import numpy as np
from functools import partial
import itertools
import signal
import time
import pdb
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
except ImportError:
    pass # or set "


# Local
from .utils import detect_kernel
from .utils import plot_2d_attrib_entailment, plot_text_attrib_entailment
from .methods import mnist_normalize, mnist_unnormalize

KERNEL=detect_kernel()

################################################################################
#####################       Scoring functions      #############################
################################################################################

# For MNIST, there are (28 - w)*(28 - h) overlapping squares to choose from
def delta_fun(hist):
    vals, inds = hist.sort(descending=True)
    return (vals[:,:-1] - vals[:,1:]).max()

def entreg_delta_fun(hist, alpha=.1):
    vals, inds = hist.sort(descending=True)
    return delta_fun(vals) - alpha*(vals - vals.log()).sum(dim=1)

def cumul_delta_fun(hist, verbose = False):
    vals, inds = hist.sort(descending=True)
    s       = torch.cumsum(vals, dim=1)
    vals, inds = hist.sort(descending=False)
    s_compl = torch.cumsum(vals, dim=1)

    #s_diff = s - s_compl.flip(dim=1) # flip in pytorch Not yet implemented https://github.com/pytorch/pytorch/pull/7873
    s_diff = s.detach().numpy()[:,:-1] - np.flip(s_compl.detach().numpy(),1)[:,1:]
    m, argm = torch.Tensor(s_diff).max(dim=1)
    if verbose:
        print(hist)
        print(s)
        print(np.flip(s_compl.detach().numpy(),1))
        print(s_diff)
        print(m, argm)
    return m

def normalized_power(hist, k = 10, alpha = .75, verbose = True):
    # P(c)
    vals, inds = hist.sort(descending=True)
    P_c        = torch.cumsum(vals[:,:-1], dim=1) #/torch.pow(torch.range(1,9),0.5)
    print('Probs:   ', P_c[0].detach().numpy())
    Cards      = torch.range(1,k-1) # We dont care about no trivial partitions l=0,k
    reg = torch.pow(torch.abs(Cards - k/2)*(2/k), 2)
    print('Reg Term:', reg.detach().numpy())
    scores = P_c - alpha*reg
    # Hack - dont want anything above k/2 card
    scores[:,int(k/2):] = 0
    print(scores)
    m, argm = scores.max(dim=1)
    C = inds[0,:argm+1]
    print(m, argm)
    if len(C) == 0:
        pdb.set_trace()
    #print(asd.asd)
    return m, C

def normalized_deltas_old(hist, k = 10, alpha = 1):
    vals, inds = hist.sort(descending=True)
    deltas  = vals[:,:-1] - vals[:,1:]
    print('Deltas:   ', deltas[0].detach().numpy())
    Cards      = torch.range(1,k-1) # We dont care about no trivial partitions l=0,k
    #reg = torch.pow(torch.abs(Cards - k/2)*(2/k), 3)
    reg = 1/torch.pow(Cards,2)
    print('Reg Term:', reg.detach().numpy())
    scores = deltas - alpha*reg
    # Hack - dont want anything above k/2 card
    scores[:,int(k/2):] = 0
    print(scores)
    m, argm = scores.max(dim=1)
    C = inds[0,:argm+1]
    print(m, argm)
    print(C)
    if len(C) == 0:
        pdb.set_trace()
    #print(asd.asd)
    return m, C

DEBUG = False
def normalized_deltas(hist, pred, alpha = 1, p = 2, reg_type='exp',
        argmin_reg = None, include_pred = True, verbose = False, **kwarg):
    """
        Computes:
    """
    true_k = len(hist)     # True ground set cardinality
    k = len(hist)

    # Values need to be sorted for delta computation
    inds = hist.argsort()[::-1]
    vals = hist[inds]

    deltas  = vals[:-1] - vals[1:]
    Cards = np.arange(1,k) # numpy and torch range have different behavior for top end of intervaal!
    if reg_type == 'exp':
        reg = 1/np.power(Cards,p)
    elif reg_type == 'quadratic':
        argmin = (k+1)/2 if argmin_reg is None else argmin_reg
        #print(argmin)
        reg = ((Cards - argmin)**p)/np.abs(argmin -1)  # Denom normalizes so that f = 1 = |c| =1
    else:
        raise ValueError('Unrecognized mode in normalized deltas')
        #reg = torch.pow(torch.abs(Cards - k/2)*(2/k), 3)

    scores = deltas - alpha*reg
    #print(scores)
    # Hack - dont want anything above k/2 card
    if k > 3:
        scores[int(k/2):] = float("-inf")
    if include_pred:
        #print(pred)
        ## index of pred class in (unsorted) hist
        pred_idx = np.where(kwarg['V'] == pred)[0][0]
        #print(pred_idx)
        ## index of pred class in sorted hist
        cut_val = np.where(inds == pred_idx)[0][0]
        #print(cut_val)
        scores[cut_val+1:] = float("-inf")
        if DEBUG:
            print('d')
            print(cut_val, scores)
            #pdb.set_trace()


    #m, argm = scores.max(dim=0)  #torc hversion
    argm = scores.argmax()
    m    = scores.max()

    C = inds[:argm+1]
    if len(C) == 0:
        pdb.set_trace()
    #print(asd.asd)
    #pdb.set_trace()
    # Map back to original indices of classes
    #C_orig = [classes[i] for i in C] if subs_k < true_k else C
    #else:
    #print('Selected classes (fake index)', C)
    #pdb.set_trace()
    return m, C


################################################################################


def ets_masker(x, S, max_len = None):
    if max_len is None:
        max_len = x.squeeze().shape[0]
    #mask = torch.zeros(1, max_len).byte() # Batch version
    mask = torch.zeros(max_len).byte() # Batch version
    mask[S] = 1
    #X_masked = torch.masked_select(x, mask).view(x.shape[0], -1) # Batch very
    X_masked = torch.masked_select(x,mask).view(x.shape[0], -1)
    return X_masked



def mnist_masker(x, S, H = 28, W = 28):
    ii = list(range(H)[S[0]])[0]
    jj = list(range(W)[S[1]])[0]
    mask = torch.zeros(H, W) # FIXME: SHOULD BE TRUE ZERO
    mask[S[0],S[1]] = 1
    X_s = mnist_unnormalize(x)*mask.view(1,1,H,W).repeat(x.shape[0], 1, 1, 1)
    X_plot  = X_s
    X_input = mnist_normalize(X_s)
    return X_input, X_plot


def image_masker_unnormalized(x, S, H = 32, W = 32):
    """
        Should work for other tasks where we don't prenormalize. Considering renaming
        to something more general.

        Since input doesn't require normalization, X_plot and X_input are the same.
    """
    ii = list(range(H)[S[0]])[0]
    jj = list(range(W)[S[1]])[0]
    mask = torch.zeros(H, W) # FIXME: SHOULD BE TRUE ZERO
    mask[S[0],S[1]] = 1
    X_s = x*mask.view(1,1,H,W).repeat(x.shape[0], 1, 1, 1)
    return X_s, X_s



################################################################################



class Explanation(object):
    def __init__(self, x, y, classes, masker = None, task = None, **kwargs):
        self.input = x
        self.prediction = y
        self.classes = classes
        self.predicates = []
        self.masker = masker # Must be passed when instantiating Explanation
        self.task   = task   # Must be passed when instantiating Explanation
        # For text tasks
        self.vocab = kwargs['vocab'] if 'vocab' in kwargs else None


    def plot(self, plot_type = 'treemap', save_path = None):
        n = len(self.predicates)
        if self.task in ['ets']:
            fig, ax = plt.subplots(n, 2, figsize = (12,4*n),
                                        gridspec_kw = {'width_ratios':[2, 1]})
        else:
            ncol = 3
            fig, ax = plt.subplots(n, 3, figsize =  (4*ncol,4*n),
                                        gridspec_kw = {'width_ratios':[1, 1, 1.3]})

        #V = np.array(list(range(len(self.classes))))

        C_prev = range(len(self.classes))
        for i, (S,C,hist_V, V_prev, V) in enumerate(self.predicates):
            row_ax = ax[i,:] if len(self.predicates) >1 else ax
            masked = self.masker(self.input, S)
            if type(masked) is tuple:
                X_s, X_s_plot = masked
            else:
                X_s = masked

            if self.task in ['hasy', 'mnist','leafsnap']:
                plot_2d_attrib_entailment(
                    self.input, X_s_plot, S, C, V_prev, hist_V, plot_type=plot_type,
                    class_names = self.classes, topk = 10, #title = labstr,
                    sort_bars = True, ax = row_ax, save_path = save_path)
            elif self.task in ['ets']:
                plot_text_attrib_entailment(self.input, X_s, S, C, V_prev, hist_V,
                                    plot_type = plot_type,
                                    class_names = self.classes,
                                    vocab = self.vocab, ax = row_ax,
                                    title = None, sort_bars = False)
        #if self.task in ['ets']:
        plt.suptitle('Prediction: {} (class {})'.format(
                            self.classes[self.prediction], self.prediction),
                            fontsize = 24, y = .95)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
        print('Done!')


class MCExplainer(object):
    def __init__(self, classifier, mask_model, classes, loss_type = 'norm_delta',
                reg_type = 'exp', crit_alpha=1, crit_p=1, plot_type = 'bar'):
        self.classifier = classifier
        self.mask_model = mask_model
        self.classes    = classes
        self.plot_type  = plot_type
        self.loss_type  = loss_type
        self.reg_type   = reg_type
        self.crit_alpha = crit_alpha
        self.crit_p     = crit_p
        self.task       = mask_model.task
        self.input_size = mask_model.input_size
        #self.input_size = mask_model.image_size # Legacy. Replace by above once new models trainerd.
        #self.pykernel   = detect_kernel()


        if self.task == 'mnist':
            self.masker = mnist_masker#, h = model.mask_size, w = model.mask_size)
            self.input_type = 'image'
        elif self.task in ['hasy', 'leafsnap']:
            H, W = self.input_size
            self.masker = partial(image_masker_unnormalized, H=H, W=W)# #partial(hasy_masker)#,  h = model.mask_size, w = model.mask_size)    )
            self.input_type = 'image'
        elif self.task == 'ets':
            self.masker = ets_masker
            self.input_type = 'text'
        else:
            raise ValueError('Unrecognized task.')
        self.criterion = self._init_criterion()


    def _init_criterion(self):

        loss = self.loss_type

        if loss == 'norm_delta':
            crit = partial(normalized_deltas, reg_type = self.reg_type,
                    alpha=self.crit_alpha, p=self.crit_p, argmin_reg = None,
                    include_pred= True)
        else:
            raise ValueError('Not implemented yet')
        #nclasses = len(V)

        # if loss == 'euclidean':
        #     # Target Histrogram
        #     target_hist = torch.zeros(1, nclasses).detach()
        #     target_hist[:,tgt_classes] = 1
        #     target_hist /= target_hist.sum()
        #     hist_loss  = (vals - target_hist).norm(p=p)
        # elif loss == 'delta':
        #     #hist_loss  = torch.abs(delta_fun(vals) - delta_fun(target_hist))
        #     hist_loss = -delta_fun(vals) # Minus because want to maximize delta
        # elif loss == 'cumul_delta':
        #     hist_loss = -cumul_delta_fun(vals, verbose = verbose) # Minus because want to maximize delta
        # elif loss == 'norm_power':
        #     obj, C = normalized_power(vals)
        #     hist_loss = -obj
        # elif loss == 'norm_delta':
        #     obj, C = normalized_deltas(vals)#, all_classes)
        #     hist_loss = -obj

        return crit

    def explain(self, x, y = None, verbose = False, show_plot = 1, save_path = None):
        # if y not provided, call classif model
        if y is None:
            out = self.classifier(x)
            p, y = out.max(1)
            y = y.item()

        if show_plot > 1 and self.input_type == 'image':
            plt.imshow(x.squeeze(), cmap='Greys')
            plt.title('Prediction: ' + self.classes[y] + ' (class no.{})'.format(y), fontsize = 20)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        else:
            print(y)
            print(self.classes)
            print('Prediction: ' + self.classes[y] + ' (class no.{})'.format(y))
            #print('hasdasd')

        V = list(range(len(self.classes)))
        E = Explanation(x, y, self.classes, masker = self.masker, task = self.task)
        step = 0
        while len(V) > 1:
            print('Explanation step: ', step)
            # if step == 5:
            #     #show_plot = 2
            #     verbose = 2
            #     global DEBUG
            #     DEBUG=True

            if verbose > 1:
                print(V)
            #pdb.set_trace()
            #max_S, max_C, hist_V = # Once attrib_optimizer is created at init with partial: self.attrib_optimizer(x, V, show_plot = show_plot, verbose = max(0, verbose-1))
            if self.task in ['mnist', 'hasy', 'leafsnap']:
                max_S, max_C, hist_V = self.optimize_over_rectangle_attributes(
                                        self.mask_model, self.criterion, self.masker,
                                        x, y, V=V, tgt_classes = [2,3,4], sort_hist = False,
                                        show_plot = show_plot, plot_type = self.plot_type,
                                        loss = self.loss_type, class_names = self.classes,
                                        force_include_class = y, verbose = max(0, verbose -1)
                                        )
            elif self.task == 'ets':
                E.vocab = self.mask_model.vocab
                max_S, max_C, hist_V = self.optimize_over_ngram_attributes(
                                        self.mask_model, self.criterion, self.masker,
                                        x, y, V=V, tgt_classes = [2,3,4], sort_hist = False,
                                        show_plot = show_plot, plot_type = self.plot_type,
                                        loss = self.loss_type, class_names = self.classes,
                                        force_include_class = y, verbose = max(0, verbose -1)
                                        )
            else:
                raise ValueError("Wrong task")
            if max_S is None:
                print('Explanation failed')
                break
            if verbose > 1:
                print('Entailed classes: ')
                print([self.classes[c] for c in max_C])

            if len(max_C) == len(V) or len(max_C) == 0:
                print('Here')
                break

            #pdb.set_trace()

            # Remove classes from reference set
            V_prev = V.copy()
            if y in max_C:
                # predicted class in entailed set - keep that
                V = [k for k in V if k in max_C]  # np version: np.isin(V, max_C)
            else:
                # predicted class not in entailed set, remove entailed set
                V = [k for k in V if k not in max_C]
            if verbose > 0:
                print('Shrinking ground set: {} -> {}'.format(V_prev, V))

            E.predicates.append((max_S, max_C, hist_V, V_prev, V))
            step += 1

        return E

    #TODO: do we need static? Maybe make just normal with self, to avoid having to pass crit function
    @staticmethod
    def optimize_over_rectangle_attributes(model, criterion, masker, x, y, V = None, tgt_classes = [1,2],
                                 sort_hist = False, show_plot = False, plot_type = 'treemap', class_names = None,
                                 loss = 'euclidean', force_include_class = None, verbose = False):
        """
            Given masking model, input x and target histogram, finds subset of x (i.e., attribute)
            that minimizes loss with respect to target historgram.

             - V: ground set of possible classes
            plot: 0/1/2 (no plotting, plot only final, plot every iter)

        """
        model.net.eval()
        H, W = model.input_size
        h, w = model.mask_size
        max_obj, max_S = float("-inf"), None
        if V is None:
            V = np.array(list(range(10)))
        elif type(V) is list:
            V = np.array(V)
        if force_include_class:
            assert force_include_class in V, "Error: class to be force-included not in ground set"

        if KERNEL == 'terminal':
            #print("PLT ION")
            fig, ax = plt.subplots(1, 3, figsize = (4*3,4))
            plt.ion()
            plt.show()
            ims = None
        else:
            fig, ax, ims = None, None, None


        def handler(*args):
            print('Ctrl-c detected: will stop iterative plotting')
            nonlocal show_plot
            show_plot = min(show_plot, 1)

        #signal.signal(signal.SIGINT, handler)
        #print(KERNEL)
        pbar = tqdm(total=(W-w)*(H-h))
        for (ii,jj) in itertools.product(range(0, W-w),range(0,H-h)):
                #print(ii,jj)
                pbar.update(1)
                S = np.s_[ii:min(ii+h,H),jj:min(jj+w,W)]
                X_s, X_s_plot = masker(x, S)
                output = model(X_s)
                hist = output.detach().numpy().squeeze()
                hist_V = hist[V]
                obj, C_rel = criterion(hist_V, pred = y, V = V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)

                # Convert from relative indices to true classes
                 #TODO: Maybe move this to criterion?
                C = V[C_rel]

                labstr = 'Current objective: {:8.4e} (i = {:2}, j = {:2})'.format(obj.item(), ii, jj)
                if show_plot > 1:
                    if KERNEL == 'ipython':
                        clear_output(wait=True)
                        ax = None

                    ax, rects, ims = plot_2d_attrib_entailment(x, X_s_plot, S, C, V, hist_V,
                                        plot_type = plot_type, class_names = class_names,
                                        title = labstr, ax = ax, ims =ims, return_elements = True
                                        )

                    if KERNEL == 'terminal':
                        plt.draw()
                        plt.pause(.005)
                        #rects[0].remove()
                        ax[0].clear() # Alternatively, do rects[0].remove(), though its slower
                        ax[1].clear()
                        ax[2].clear()
                    else:
                        plt.show()
                elif verbose:
                    print(labstr)

                if force_include_class is not None and (not force_include_class in C):
                    # Target class not in entailed set
                    #print(force_include_class, C)
                    #print(obj, C_rel)
                    #global DEBUG
                    #DEBUG = True
                    #obj, C_rel = criterion(hist_V, pred = y, V = V)#, verbose = True)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
                    obj = float("-inf")
                    #pdb.set_trace()
                    continue
                if obj > max_obj:
                    max_obj = obj
                    max_S   = S

        pbar.close()
        if max_S is None:
            print('Warning: could not find any attribute that causes the predicted class be entailed')
            return None, None, None

        if KERNEL == 'terminal':
            plt.ioff()
            plt.close('all')

        # Reconstruct vars for optimal
        X_s, X_s_plot = masker(x, max_S)
        output = model(X_s)
        hist = output.detach().numpy().squeeze()
        hist_V = hist[V]
        obj, C_rel = criterion(hist_V, pred = y, V=V)#, sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
        max_C = V[C_rel]

        ii, jj =  list(range(H)[max_S[0]])[0], list(range(W)[max_S[1]])[0]

        assert obj == max_obj
        labstr = 'Best objective: {:8.4f} (i = {:2}, j = {:2})'.format(max_obj.item(), ii, jj)
        #clear_output(wait=True)
        if show_plot:
            plot_2d_attrib_entailment(x, X_s_plot, max_S, max_C, V, hist_V, plot_type=plot_type,
            class_names = class_names,
            topk = 10, title = labstr, sort_bars = True)
            plt.show()
        else:
            print(labstr)

        return max_S, max_C, hist_V

    @staticmethod
    def optimize_over_ngram_attributes(model, criterion, masker, x, y, V = None,
                                 tgt_classes = [1,2], sort_hist = False,
                                 show_plot = False, plot_type = 'treemap',
                                 class_names = None, loss = 'euclidean',
                                 force_include_class = None, verbose = False):
        """
            Given masking model, input x and target histogram, finds subset of x
            (i.e., attribute) that minimizes loss with respect to target historgram.

             - V: ground set of possible classes
            plot: 0/1/2 (no plotting, plot only final, plot every iter)

        """
        model.net.eval()
        N = x.shape[1]
        n = model.mask_size
        max_obj, max_S = 100*torch.ones(1), None
        if V is None:
            V = np.array(list(range(10)))
        elif type(V) is list:
            V = np.array(V)
        if force_include_class:
            assert force_include_class in V, "Error: class to be force-included not in ground set"

        if KERNEL == 'terminal':
            fig, ax = plt.subplots(1, 2, figsize = (10,4),
                                gridspec_kw = {'width_ratios':[2, 1]})
            plt.ion()
            plt.show()
            ims = None
        else:
            fig, ax, ims = None, None, None

        def handler(*args):
            print('Ctrl-c detected: will stop iterative plotting')
            nonlocal show_plot
            show_plot = min(show_plot, 1)

        signal.signal(signal.SIGINT, handler)

        for ii in range(0, N-n):
            #print(ii)
            S = np.s_[range(ii,ii+n)]
            X_s    = masker(x, S)
            output = model(X_s)
            hist = output.detach().numpy().squeeze()
            hist_V = hist[V]
            obj, C_rel = criterion(hist_V, pred = y, V = V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
            #obj, C_rel = criterion(hist_V, V=V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
            # Convert from relative indices to true classes
             #TODO: Maybe move this to criterion?
            C = V[C_rel]
            labstr = 'Ngram: [{}:{}], Objective: {:8.4e}'.format(ii, ii+n,obj.item())
            ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
            if show_plot > 1:
                if KERNEL == 'ipython':
                    clear_output(wait=True)
                    ax = None

                ax = plot_text_attrib_entailment(x, X_s, S, C, V, hist_V,
                                    plot_type = plot_type,
                                    class_names = class_names,
                                    vocab = model.vocab, ax = ax,
                                    title = labstr, sort_bars = False)
                if KERNEL == 'terminal':
                    plt.draw()
                    plt.pause(.005)
                    ax[0].clear()
                    ax[1].clear()
                else:
                    plt.show()
            elif verbose:
                #ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
                #labstr = 'Ngram: {} ([{}:{}]), Objective: {:8.4e}'.format(ngram_str, ii, ii+n,obj.item())
                print(labstr)

            if force_include_class is not None and (not force_include_class in C):
                # Target class not in entailed set
                #print(force_include_class, C)
                obj = float("-inf")
                continue
            if obj < max_obj:
                max_obj = obj
                max_S   = S
                #clear_output(wait=True)

        if max_S is None:
            print('Warning: could not find any attribute that causes to predicted class be entailed')
            return None, None, None

        if KERNEL == 'terminal':
            plt.ioff()
            plt.close('all')

        # Reconstruct vars for optimal
        X_s = masker(x, max_S)
        output = model(X_s)
        hist = output.detach().numpy().squeeze()
        hist_V = hist[V]
        obj, C_rel = criterion(hist_V, pred = y, V = V)#, sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
        max_C = V[C_rel]
        #print(asd.asd)
        ii = max_S[0]

        assert obj == max_obj
        ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
        labstr = 'Best objective: {:8.4f}, (i = {:2})'.format(max_obj.item(),ii)
        #clear_output(wait=True)
        if show_plot:
                plot_text_attrib_entailment(x, X_s, max_S, max_C, V, hist_V,
                                    plot_type = plot_type,
                                    class_names = class_names,
                                    vocab = model.vocab,
                                    title = labstr, sort_bars = False)
                plt.tight_layout()
                plt.show()
        else:
            print(labstr)
        return max_S, max_C, hist_V

    # def criterion(self, pred, V, tgt_classes = [7,9], sort_hist = False, p = 1, loss = 'delta', verbose = False):
    #     nclasses = len(V)
    #     if sort_hist:
    #         vals, inds = pred.sort(descending=True)
    #     else:
    #         vals = pred
    #     if loss == 'euclidean':
    #         # Target Histrogram
    #         target_hist = torch.zeros(1, nclasses).detach()
    #         target_hist[:,tgt_classes] = 1
    #         target_hist /= target_hist.sum()
    #         hist_loss  = (vals - target_hist).norm(p=p)
    #     elif loss == 'delta':
    #         #hist_loss  = torch.abs(delta_fun(vals) - delta_fun(target_hist))
    #         hist_loss = -delta_fun(vals) # Minus because want to maximize delta
    #     elif loss == 'cumul_delta':
    #         hist_loss = -cumul_delta_fun(vals, verbose = verbose) # Minus because want to maximize delta
    #     elif loss == 'norm_power':
    #         obj, C = normalized_power(vals)
    #         hist_loss = -obj
    #     elif loss == 'norm_delta':
    #         obj, C = normalized_deltas(vals)#, all_classes)
    #         hist_loss = -obj
    #     return hist_loss, C
