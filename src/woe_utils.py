import os, sys
from attrdict import AttrDict
import argparse
import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

DEBUG = False

# sys.path.append(os.path.expandvars('$HOME')+'/workspace/normalizing_flows')
# from maf import MAF

def mnist_unnormalize(x):
    return x.clone().mul_(.3081).add_(.1307)

def mnist_normalize(x):
    return x.clone().sub_(.1307).div_(.3081)


def logit(x, eps=1e-5):
    x.clamp_(eps, 1 - eps)
    return x.log() - (1 - x).log()

def invlogit(x):
    """
        Elementwise inverse logit (a.k.a logistic sigmoid.)
        :param x: numpy array
        :return: numpy array
    """
    # alternatively:  1.0 / (1.0 + torch.exp(-x)
    return x.exp()/(1+x.exp())

def int2hot(idx, n = None):
    if type(idx) is int:
        idx = torch.tensor([idx])
    if idx.dim == 0:
        idx = idx.view(-1,1)
    if not n:
        n = idx.max() + 1
    return torch.zeros(len(idx),n).scatter_(1, idx.unsqueeze(1), 1.)

# def one_hot(x, label_size):
#     out = torch.zeros(len(x), label_size).to(x.device)
#     out[torch.arange(len(x)), x] = 1
#     return out

#### Should try to make woe_wrapper parent class agnostic to torch/numpy/scikit.

class woe_wrapper():
    def __init__(self, model, task, classes, input_size):
        self.model = model
        self.classes = classes
        self.task  = task
        self.input_size = input_size
        self.data_transform = None
        self.cache = None
        self.caching = False

    def _start_caching(self):
        self.caching = True
        self.cache = {}

    def _stop_caching(self):
        self.caching = False
        self.cache = None

    def woe(self, x, y1, y2, **kwargs):
        #ll_num   = self.log_prob(x, y1, args)
        #ll_denom = self.log_prob(x, y2, args)
        ll_num   = self.generalized_conditional_ll(x, y1, **kwargs)
        ll_denom = self.generalized_conditional_ll(x, y2, **kwargs)
        woe = ll_num - ll_denom
        return woe

    def generalized_conditional_ll(self, x, y, subset=None, verbose = False, **kwargs):
        """A wrapper to handle different cases for conditional LL computation:
        simple or complex hypothesis.

        Parameters
        ----------
        model : torch model
            Model to evaluate
        x : array-like, shape = [n_subset]
            Input
        y : int or list of ints
            If just an int, this is simpl hypthesis ratio, if list it's composite
        Returns
        -------
        log probability : float
        """
        k = len(self.classes)

        # We memoize woe computation for efficiency. For this, need determinsitc hash
        # This has to be done before doing view or reshape on x.
        hash_x = id(x) #hash(x)  - hash is non determinsitc!
        hash_S = -1 if subset is None else subset[0].item()
        if DEBUG and hash_S != -1: print('here', hash_S)

        x = x.view(x.shape[0], -1) #.to(args.device)
        #pdb.set_trace()
        if type(y) == set:
            y = list(y)
        if type(y) is int or (type(y) == list and len(y) == 1):
            if verbose: print('H is simple hypotesis')
            y = y if type(y) is list else [y]
            y = torch.tensor(y*x.shape[0])
            y = int2hot(y, k)
            if subset is None:
                loglike = self.log_prob(x, y)
            else:
                #pdb.set_trace()
                loglike = self.log_prob_partial(subset, x, y)
        elif type(y) is list:
            if verbose: print('H is composite hypotesis')
            priors = (torch.ones(k)/k)#.to(args.device) ## TODO: REAL PRIORS!!
            logpriors = torch.log(priors[y])
            logprior_set = torch.log(priors.sum())
            loglike = [[] for _ in range(len(y))]

            for i, yi in enumerate(y):
                #pdb.set_trace()
                if self.caching and ((hash_x,hash_S)in self.cache) and (yi in self.cache[(hash_x,hash_S)]):
                    if DEBUG and hash_S != -1: print('using cached: ({},{})'.format(hash_x, hash_S))
                    loglike[i] = self.cache[(hash_x,hash_S)][yi]
                else:
                    yi_hot = int2hot(torch.tensor([yi]*x.shape[0]), k)
                    if subset is None:
                        loglike[i] = self.log_prob(x, yi_hot) + logpriors[i] - logprior_set
                    else:
                        loglike[i] = self.log_prob_partial(subset, x, yi_hot) + logpriors[i] - logprior_set
                    if self.caching:
                        if not (hash_x,hash_S) in self.cache:
                            self.cache[(hash_x,hash_S)] = {}
                        self.cache[(hash_x,hash_S)][yi] = loglike[i]

            #pdb.set_trace()
            loglike = torch.stack(loglike, dim=1)
            # log p(x|Y∈C) = 1/P(Y∈C) * log ∑_y p(x|Y=y)p(Y=y)
            #              = log ∑_y exp{ log p(x|Y=y) + log p(Y=y) - log P(Y∈C)}
            loglike = torch.logsumexp(loglike, dim = 1)
        return loglike

    def decomposition_woe(self, x, y1, y2, exid = None, plot = True, rgba= True, order = 'fwd', figsize = (10,10)):
        """
            Note that regardless or display and computation order, will return partial
            woes in "natural" order (L->R, T->D)
        """
        # TODO get these number from model attribs
        k = 7
        d = int(np.sqrt(self.input_size))#[0]
        input_order_idxs, block_order_idxs = get_block_order(7, d, order = order)

        # if order == 'fwd':
        #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= False)
        # elif order == 'bwd':
        #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= True)
        # elif order == 'rnd':
        #     order, blockord = randperm_block_order(get_block_order(7, int(np.sqrt(input_size))), 7)
        # else:
        #     raise ValueError("Unrecognized order")


        total_woe = self.woe(x, y1, y2)#,  args)

        nblocks = d**2 // k**2
        nrowb = ncolb = int(np.sqrt(nblocks))

        partial_woes = torch.zeros(x.shape[0], nblocks)

        if plot:
            # If no specific index provided We'll display the examlpe with the max total woe
            if x.shape[0] == 1:
                exid = 0
            elif exid is None:
                max_woe, exid = total_woe.max(0)
            print(exid)
            if rgba:
                plt.imshow(invlogit(x[exid,:]).reshape(28,28))
            else:
                plt.imshow(invlogit(x[exid,:]).reshape(28,28), cmap = 'Greys')
            plt.title('Total Woe: {:8.2f}'.format(total_woe[exid]))
            plt.show()


        ll_num_prev = ll_denom_prev = torch.zeros_like(x[:,0]) - 10000000000

        attrib_idxs = []
        for nb in range(nblocks):
            # nb is sequential order, bidx is "true" block index, in LR TD order
            # these are equal for order == 'fwd', but different in other cases
            bidx = block_order_idxs[nb]
            I  = nb // (28//k)
            J  = nb % (28//k)
            init_idx = nb*(k**2)
            block_idxs = input_order_idxs[init_idx:init_idx+k**2] # Input indices for this block

            cumulative = True
            #print(block_idxs)
            ### DEBUG
            #cumul_block_idxs = input_order_idxs[:init_idx+k**2]
            if cumulative:
                ll_num   = self.generalized_conditional_ll(x, y1, block_idxs)
                ll_denom = self.generalized_conditional_ll(x, y2, block_idxs)
            else:
                ll_num   = (self.generalized_conditional_ll(x, y1, block_idxs).exp() - ll_num_prev.exp()).log()
                ll_denom = (self.generalized_conditional_ll(x, y2, block_idxs).exp() - ll_denom_prev.exp()).log()
                ll_num_prev = ll_num
                ll_denom_prev = ll_denom

            woe = ll_num - ll_denom
            partial_woes[:, bidx] = woe
            attrib_idxs.append(block_idxs)

        partial_woes = partial_woes.detach().numpy()
        total_woe = total_woe.detach().numpy()

        if plot:
            self.plot_partial_woe(x[exid,:].squeeze(), partial_woes[exid,:].squeeze(), k, d, order)


        return total_woe, partial_woes, attrib_idxs #print(partial_woes.sum(axis=1).shape)

    @staticmethod
    def plot_partial_woe(x, partial_woes, block_width, input_width, order, rgba=False, figsize = (10,10)):
        if x.ndimension() > 1:
            x = x.view(-1) #.to(args.device)
        if rgba:
            cmap = plt.get_cmap()
            full_img = cmap(x)
            partial_img = 0*full_img
        else:
            full_img = x
            partial_img = full_img*0 + .1

        input_order_idxs, block_order_idxs = get_block_order(block_width, input_width, order = order)
        nblocks = input_width**2 // block_width**2
        nrowb = ncolb = int(np.sqrt(nblocks))

        plt.figure(figsize = figsize)
        gs1 = gridspec.GridSpec(nrowb, ncolb)
        gs1.update(wspace=0.00, hspace=0.2) # set the spacing between axes.

        #pdb.set_trace()

        woe_ranks = sp.stats.rankdata(-partial_woes, method = 'ordinal')
        for nb in range(nblocks):
            bidx = block_order_idxs[nb]
            I  = nb // (28//block_width)
            J  = nb % (28//block_width)
            init_idx = nb*(block_width**2)
            block_idxs = input_order_idxs[init_idx:init_idx+block_width**2]
            #ax = plt.subplot(gs1[nb]) if not reverse else plt.subplot(gs1[nblocks - nb - 1])
            ax = plt.subplot(gs1[bidx]) # Plot panes will have natural order hence bidx not nb
            partial_img[block_idxs] = full_img[block_idxs]
            if rgba:
                ax.imshow(partial_img.reshape(28,28,4))
            else:
                ax.imshow(partial_img.reshape(28,28), cmap = 'Greys', vmin = 0, vmax = 1)
            plt.axis('on')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            ii, jj = np.unravel_index(min(block_idxs), (28,28))
            w, h   = block_width, block_width
            rects = patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rects)
            title = '{:2.2f}'.format(partial_woes[bidx])
            if partial_woes[bidx] > 10:
                title = r"$\bf{" + title + "}$"
            ax.set_title('Woe ' + title + ' ({}/{})'.format(woe_ranks[bidx], nblocks))

            # Grey out these pixels.
            if rgba:
                partial_img[block_idxs,3] *= 0.5
            else:
                partial_img[block_idxs] *= 0.05

        plt.show()


class woe_maf(woe_wrapper):
    """
        Wrapper around MAF models to compute WOE from woe.py methods easily and agnostically.

        Ideally, WOE should be completely model agnostic. And just expect most
        WOE computation-related funs to be implemented in the classes. Ideally WOE_MAF
        and other wrappers should have methods:
            - log_prob(x, y, [subset]) which take both y int or list. (would substitiue generalized_conditional_ll)

            - maybe also woe (now called get_woe), though it's so simple that not sure it's worh it.
              Perhaps include that within parent woe_wrapper class.


    """
    def __init__(self, model, task, classes, input_size):
        super().__init__(model, task, classes, input_size)
        #self.model = model
        # For caching and efficiency
        self.x     = None
        self.h1    = None
        self.h2    = None
        self.u_h1     = None
        self.log_jacobian_h1 = None

    def forward(self, x, y):
        # MAF models expect x to be B X (D^2) and Y to be one hot B x NCLASSES
        if x.ndimension() > 2:
            x = x.reshape(x.shape[0], -1)
        if y.ndimension() == 1:
            y = int2hot(y, len(self.classes))

        if self.data_transform is not None:
            x = self.data_transform(x.clone()) ## Read that torch.transforms are inplace, so avoid modifying permnentlyu
        #print(x.min())
        #print(x, y)
        # self.hash(x)
        return self.model.forward(x, y)

    def log_prob(self, x, y):
        """
            Returns conditional log probability P(X|Y).

            Identical to original MAF function but calls the wrapper's forward methjod instead
        """
        u, log_abs_det_jacobian = self.forward(x, y)
        ## TODO: I HAD self.u, self.log_abs_det_jacobian = self.forward(x, y). Really need to store?
        log_probs_base = self.model.base_dist.log_prob(u)
        return torch.sum(log_probs_base + log_abs_det_jacobian, dim=1)

    ## THIS HAS NOW BEEN ADDED AS METHOD OF MADE. Eliminate from here eventually
    def log_prob_partial(self, entries, x, y=None):
        """Compute partial log probability of autoregressive model
        Parameters
        ----------
        model : torch model
            Autoeregressive model to evaluate
        entries : array-like, shape = [n_subset]
            Subset of indices that will go into loprob compoutation
        x : int
            Rank.
        Returns
        -------
        log probability : float
        """
        #pdb.set_trace()
        u, log_abs_det_jacobian = self.forward(x, y)
        log_probs_base = self.model.base_dist.log_prob(u)[:,entries]
        #lp2 = model.base_dist.log_prob(u[:,entries]) # Doesnt work
        log_abs_det_jacobian = log_abs_det_jacobian[:,entries]
        return torch.sum(log_probs_base + log_abs_det_jacobian, dim=1)

    def partial_conditional_logprob_deprecated(self, x, y, entries):
        """ TODO: What was is coding this for?
        """
        if x != self.x:
            self.u = None
            self.log_abs_det_jacobian = None


        self.u, self.log_abs_det_jacobian = self.model.forward(x, y)

        log_probs_base = self.model.base_dist.log_prob(u)[:,entries]
        #lp2 = model.base_dist.log_prob(u[:,entries]) # Doesnt work
        log_abs_det_jacobian = log_abs_det_jacobian[:,entries]






def randperm_block_order(order, k):
    # Randomize order in which blocks are traversed, while keep consistent LtoR UtoD
    # order within each of these
    n = len(order)
    nblocks = n//k**2
    bsize   = k**2
    blockord = torch.randperm(nblocks)
    permorder = torch.empty_like(order)
    #pdb.set_trace()
    for i in range(nblocks):
        j = blockord[i].item()
        #print('{}-th block is {}th'.format(i,j))
        permorder[i*bsize:(i+1)*bsize]  = order[j*bsize:(j+1)*bsize]
    #pdb.set_trace()
    return permorder, blockord

def get_block_order(k, n, order = 'fwd'):
    # This assumes squared input n x n and squared attribute k x k
    # n: input dimension
    # k: attribute dim
    # order can be one of 'fwd', 'bwd', 'rnd' or a list of indices for blocks
    # TODO: fold randperm block order into this function
    IDX = np.zeros((n,n), dtype = int)
    nblocks = n**2//k**2
    if type(order) is list:
        assert nblocks == len(list)

    for i in range(n) :
        for j in range(n):
            I  = i // k
            J  = j // k
            ip = i % k # internal indexing within this block
            jp = j % k
            IDX[i,j] =  ((n/k)*I + J)*(k**2) + (ip*k + jp)
    #print(IDX)
    idx2block = dict(zip(range(n**2), IDX.flatten()))
    input_idxs = torch.tensor(sorted(idx2block, key=idx2block.get))

    if order == 'fwd':
        block_idxs = range(nblocks)
    if order == 'bwd':
        input_idxs = torch.flip(input_idxs, [0])
        block_idxs = range(nblocks)[::-1]
    elif order == 'rnd':
        input_idxs, block_idxs = randperm_block_order(input_idxs, 7)

    #print(IDX[0:k,0:k]) # First Block
    #print(IDX[-k:,-k:]) # Last Block
    return input_idxs, block_idxs

# ## THIS HAS NOW BEEN ADDED AS METHOD OF MADE. Eliminate from here eventually
# def log_prob_partial(model, entries, x, y=None):
#     """Compute partial log probability of autoregressive model
#     Parameters
#     ----------
#     model : torch model
#         Autoeregressive model to evaluate
#     entries : array-like, shape = [n_subset]
#         Subset of indices that will go into loprob compoutation
#     x : int
#         Rank.
#     Returns
#     -------
#     log probability : float
#     """
#     u, log_abs_det_jacobian = model.forward(x, y)
#     log_probs_base = model.base_dist.log_prob(u)[:,entries]
#     #lp2 = model.base_dist.log_prob(u[:,entries]) # Doesnt work
#     log_abs_det_jacobian = log_abs_det_jacobian[:,entries]
#     return torch.sum(log_probs_base + log_abs_det_jacobian, dim=1)

# model
def make_model(args):
    if args.model == 'made':
        model = MADE(args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'mademog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MADEMOG(args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'maf':
        model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model == 'mafmog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MAFMOG(args.n_blocks, args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model =='realnvp':
        model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        batch_norm=not args.no_batch_norm)
    else:
        raise ValueError('Unrecognized model.')
    return model


#### THIS IS NOT WOE. THIS IS MAF, should probaly go to mad or made more general
def restore_model_from_file(model_file):

    try:
        state = torch.load(model_file, map_location='cpu')
        args = state['args']
    except:
        # Older model that doesn't save args. Will have to infer them.
        print('Saved model doesnt have args')
        args = AttrDict({
            'dataset': 'MNIST',
            'model': 'maf',
            #'restore_file':
            #'n_blocks': 8 if large_model else 5,
            'flip_toy_var_order': False,
            'input_size': 784,
            'input_dims': (1, 28, 28),
            'cond_label_size': 10,
            #'n_hidden': 5 if large_model else 3,
            'activation_fn': 'relu',
            #'input_order': 'blocks',
            'conditional': True,
            'no_batch_norm': False,
            'batch_size': 100,
            'lr': 1e-4,
        })

        # Hacky
        splitted = [s.replace('B','').replace('H','').replace('LR','') for s in os.path.dirname(model_file).split('/')[-1].split('_')]
        if len(splitted) == 3:
            B,H,HS = splitted
        else:
            B,H,HS,LR = splitted

        args.n_blocks = int(B)
        args.n_hidden = int(H)
        args.hidden_size = int(HS)

        input_order = os.path.dirname(model_file).split('/')[-2] #.split('_')[:1]
        if input_order == "blocks_fwd":
            input_order = "blocks"
        elif input_order == "blocks_bwd":
            input_order = "blocks-reverse"
        elif input_order == "blocks_rnd":
            input_order = "blocks-random"
        args.input_order = input_order


    #args.restore_file = model_file
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(args)
    model = make_model(args)
    model = model.to(args.device)

    # load model and optimizer states
    state = torch.load(model_file, map_location=args.device)
    model.load_state_dict(state['model_state'])
    #args.output_dir = os.path.dirname(args.restore_file)
    #print(model.eval())
    model.eval()
    print("Loaded model from epoch {} succesfully".format(state['epoch']))
    return model, args


parser = argparse.ArgumentParser()
parser.add_argument('model_a', type=str, help='Train a flow.')
parser.add_argument('model_b', type=str, help='Train a flow.')

# action
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
# parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
# parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
# parser.add_argument('--data_dir', default='./data/', help='Location of datasets.')
# parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
# parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
# parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# # data
parser.add_argument('--dataset', default='toy', help='Which dataset to use.')
# parser.add_argument('--flip_toy_var_order', action='store_true', help='Whether to flip the toy dataset variable order to (x2, x1).')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')
# # model
# parser.add_argument('--model', default='maf', help='Which model to use: made, maf.')
# # made parameters
# parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
# parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
# parser.add_argument('--hidden_size', type=int, default=100, help='Hidden layer size for MADE (and each MADE block in an MAF).')
# parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
# parser.add_argument('--activation_fn', type=str, default='relu', help='What activation function to use in the MADEs.')
# parser.add_argument('--input_order', type=str, default='sequential', help='What input order to use (sequential | random).')
# parser.add_argument('--conditional', default=False, action='store_true', help='Whether to use a conditional model.')
# parser.add_argument('--no_batch_norm', action='store_true')
# # training params
# parser.add_argument('--batch_size', type=int, default=100)
# parser.add_argument('--n_epochs', type=int, default=50)
# parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
# parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
# parser.add_argument('--log_interval', type=int, default=1000, help='How often to show loss statistics and save samples.')
#

import pdb


if __name__ == '__main__':

    args = parser.parse_args()
    path1 = 'results/maf/blocks_fwd/B5_H3_1024/'
    path2 = 'results/maf/blocks_bwd/B5_H3_1024/'
    #pdb.set_trace()

    # setup device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)


    for i in range(2):
        state = torch.load(args.model_a, map_location=args.device)
        pdb.set_trace()
        model_args = state.args


    #

    # load data
    if args.conditional: assert args.dataset in ['MNIST', 'CIFAR10'], 'Conditional inputs only available for labeled datasets MNIST and CIFAR10.'
    train_dataloader, test_dataloader = fetch_dataloaders(args.dataset, args.batch_size, args.device, args.flip_toy_var_order)
    args.input_size = train_dataloader.dataset.input_size
    args.input_dims = train_dataloader.dataset.input_dims
    args.cond_label_size = train_dataloader.dataset.label_size if args.conditional else None

    # model
    if args.model == 'made':
        model = MADE(args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'mademog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MADEMOG(args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'maf':
        model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model == 'mafmog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MAFMOG(args.n_blocks, args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model =='realnvp':
        model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        batch_norm=not args.no_batch_norm)
    else:
        raise ValueError('Unrecognized model.')

    print(args)
    #pdb.set_trace()
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    if args.restore_file:
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        args.start_epoch = state['epoch'] + 1
        # set up paths
        args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)
