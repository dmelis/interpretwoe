import matplotlib.pyplot as plt
import numpy as np
import pdb
DEBUG = False



#### Should try to make woe_wrapper parent class agnostic to torch/numpy/scikit.
class woe_wrapper():
    """
        Child methods should specify the following methods:
            - log_prob
            - log_prob_partial
            - prior_odds
    """
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
        ll_num   = self.generalized_conditional_ll(x, y1, **kwargs)
        ll_denom = self.generalized_conditional_ll(x, y2, **kwargs)
        woe = ll_num - ll_denom
        return woe

    def generalized_conditional_ll(self, x, y, subset=None, verbose = False, **kwargs):
        """A wrapper to handle different cases for conditional LL computation:
        simple or complex hypothesis.

        Parameters
        ----------
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
        hash_S = -1 if subset is None else subset[0]#.item()
        if DEBUG and hash_S != -1: print('here', hash_S)

        #x = x.view(x.shape[0], -1) #.to(args.device) #TORCH
        #x =

        if type(y) == set:
            y = list(y)
        if type(y) is int or (type(y) == list and len(y) == 1):
            if verbose: print('H is simple hypotesis')
            y = y if type(y) is list else [y]
            #y = torch.tensor(y*x.shape[0]) #TORCH
            #y = int2hot(y, k) # TORCH
            if subset is None:
                loglike = self.log_prob(x, y)
            else:
                loglike = self.log_prob_partial(subset, x, y)
        elif type(y) is list:
            if verbose: print('H is composite hypotesis')
            #priors = (torch.ones(k)/k)#.to(args.device) ## TODO: REAL PRIORS!!
            priors    = np.ones(k)/k
            logpriors = np.log(priors[y])
            logprior_set = np.log(priors.sum())
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

class woe_scikit_gnb(woe_wrapper):
    """
        WOE Wrapper around scikit GNB classifier
    """
    def __init__(self, clf, task=None, classes=None, input_size=None):
        super().__init__(clf, task, classes, input_size)
        self.cond_type = 'nb'

    def log_prob(self, x, y):
        """
        """
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.model.sigma_[y, :]))
        n_ij -= 0.5 * np.sum(((x - self.model.theta_[y, :]) ** 2) /
                         (self.model.sigma_[y,:]), 1)
        return n_ij

    def log_prob_partial(self, subset, x, y):
        # Using naive Bayes, we do P(X_s | X_s1,...X_s, Y) = P (X_s, Y)
        # for GNB, this is easy: just drop non-included vars
        # en.wikipedia.org/wiki/Multivariate_normal_distribution#Marginal_distributions
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.model.sigma_[y, subset]))
        n_ij -= 0.5 * np.sum(((x[:,subset] - self.model.theta_[y, subset]) ** 2) /
                         (self.model.sigma_[y, subset]), 1)
        return n_ij

    def prior_odds(self, y_num, y_den):
        return np.log(self.model.class_prior_[y_num]/self.model.class_prior_[y_den])
