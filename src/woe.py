import numpy as np
import torch
import scipy as sp
import scipy.stats
import pdb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

from .woe_utils import int2hot, log_prob_partial, get_block_order

def logistic(x):
    """
    Elementwise logistic sigmoid.
    :param x: numpy array
    :return: numpy array
    """
    return 1.0 / (1.0 + np.exp(-x))


def generalized_conditional_ll(model, x, y, args, subset=None, verbose = False):
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
    x = x.view(x.shape[0], -1).to(args.device)
    if type(y) is int:
        if verbose: print('H is simple hypotesis')
        y = torch.tensor([y]*x.shape[0])
        y = int2hot(y, args.cond_label_size)
        if subset is None:
            loglike = model.log_prob(x, y)
        else:
            loglike = log_prob_partial(model, subset, x, y)
    elif type(y) is list:
        if verbose: print('H is composite hypotesis')
        priors = (torch.ones(args.cond_label_size)/args.cond_label_size).to(args.device)
        logpriors = torch.log(priors[y])
        logprior_set = torch.log(priors.sum())
        loglike = [[] for _ in range(len(y))]

        for i, yi in enumerate(y):
            yi = torch.tensor([yi]*x.shape[0])
            yi = int2hot(yi, args.cond_label_size)
            if subset is None:
                loglike[i] = model.log_prob(x, yi) + logpriors[i] - logprior_set
            else:
                loglike[i] = log_prob_partial(model, subset, x, yi) + logpriors[i] - logprior_set

        loglike = torch.stack(loglike, dim=1)

        # log p(x|Y∈C) = 1/P(Y∈C) * log ∑_y p(x|Y=y)p(Y=y)
        #              = log ∑_y exp{ log p(x|Y=y) + log p(Y=y) - log P(Y∈C)}

        loglike = torch.logsumexp(loglike, dim = 1)

    return loglike



def get_woe(model, x, y1, y2, args):
    """
        Computes full model WOE
    """
    ll_num   = generalized_conditional_ll(model, x, y1, args)
    ll_denom = generalized_conditional_ll(model, x, y2, args)
    woe = ll_num - ll_denom
    return woe


def decomposition_woe(model, x, y1, y2, args, exid = None, plot = True, rgba= True, order = 'fwd', figsize = (10,10)):
    """
        Note that regardless or display and computation order, will return partial
        woes in "natural" order (L->R, T->D)
    """
    # TODO get these number from model attribs
    k = 7
    input_size = 28*28

    input_order_idxs, block_order_idxs = get_block_order(7, int(np.sqrt(input_size)), order = order)

    # if order == 'fwd':
    #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= False)
    # elif order == 'bwd':
    #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= True)
    # elif order == 'rnd':
    #     order, blockord = randperm_block_order(get_block_order(7, int(np.sqrt(input_size))), 7)
    # else:
    #     raise ValueError("Unrecognized order")


    total_woe = get_woe(model, x, y1, y2,  args)


    nblocks = input_size // k**2
    nrowb = ncolb = int(np.sqrt(nblocks))

    partial_woes = torch.zeros(x.shape[0], nblocks)

    if plot:
        # If no specific index provided We'll display the examlpe with the max total woe
        if exid is None:
            max_woe, exid = total_woe.max(0)
        print(exid)
        if rgba:
            plt.imshow(logistic(x[exid,:]).reshape(28,28))
        else:
            plt.imshow(logistic(x[exid,:]).reshape(28,28), cmap = 'Greys')
        plt.title('Total Woe: {:8.2f}'.format(total_woe[exid]))
        plt.show()

        if rgba:
            cmap = plt.get_cmap()
            full_img = cmap(logistic(x[exid,:]))
            partial_img = 0*full_img
        else:
            full_img = logistic(x[exid,:])
            partial_img = full_img*0 + .1


    ll_num_prev = ll_denom_prev = torch.zeros_like(x[:,0]) - 10000000000

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
            ll_num   = generalized_conditional_ll(model, x, y1, args, block_idxs)
            ll_denom = generalized_conditional_ll(model, x, y2, args, block_idxs)
        else:
            ll_num   = (generalized_conditional_ll(model, x, y1, args, block_idxs).exp() - ll_num_prev.exp()).log()
            ll_denom = (generalized_conditional_ll(model, x, y2, args, block_idxs).exp() - ll_denom_prev.exp()).log()
            ll_num_prev = ll_num
            ll_denom_prev = ll_denom

        woe = ll_num - ll_denom
        partial_woes[:, bidx] = woe

    if plot:
        plt.figure(figsize = figsize)
        gs1 = gridspec.GridSpec(nrowb, ncolb)
        gs1.update(wspace=0.00, hspace=0.2) # set the spacing between axes.
        woe_ranks = sp.stats.rankdata(-partial_woes[exid,:].detach().numpy(), method = 'ordinal')
        for nb in range(nblocks):
            bidx = block_order_idxs[nb]
            I  = nb // (28//k)
            J  = nb % (28//k)
            init_idx = nb*(k**2)
            block_idxs = input_order_idxs[init_idx:init_idx+k**2]
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
            w, h   = k, k
            rects = patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rects)
            title = '{:2.2f}'.format(partial_woes[exid,bidx])
            if partial_woes[exid,bidx] > 10:
                title = r"$\bf{" + title + "}$"
            ax.set_title('Woe ' + title + ' ({}/{})'.format(woe_ranks[bidx], nblocks))

        # Grey out these pixels.
        if rgba:
            partial_img[block_idxs,3] *= 0.5
        else:
            partial_img[block_idxs] *= 0.25

        plt.show()


    partial_woes = partial_woes.detach().numpy()
    total_woe = total_woe.detach().numpy()
    # if order == 'bwd':
    #     partial_woes = np.fliplr(partial_woes)
    # elif order == 'rnd':
    #     partial_woes = [partial_woes[i] for i in block_order_idxs]

    return total_woe, partial_woes #print(partial_woes.sum(axis=1).shape)



def decomposition_woe_efficient(model, x, y1, y2, args, exid = None, plot = True, rgba= True, order = 'fwd', figsize = (10,10)):
    """
        TODO: Finish this. Maybe make a class woe, have u log_abs_det_jacobian as attribs,
        compute first time and store there.
        But, perhaps this is too MAF dependent to be here. Maybe need a wrapper class
        around MAF to add these functionalities and have woe methods simply call their LL method.
        Same as above, but avoids running model forward method multiple times.
        Uglier but faster.
    """
    # TODO get these number from model attribs
    k = 7
    input_size = 28*28

    input_order_idxs, block_order_idxs = get_block_order(7, int(np.sqrt(input_size)), order = order)

    # if order == 'fwd':
    #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= False)
    # elif order == 'bwd':
    #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= True)
    # elif order == 'rnd':
    #     order, blockord = randperm_block_order(get_block_order(7, int(np.sqrt(input_size))), 7)
    # else:
    #     raise ValueError("Unrecognized order")

    u, log_abs_det_jacobian = model.forward(x, y)
    log_probs_base = model.base_dist.log_prob(u)[:,entries]
    #lp2 = model.base_dist.log_prob(u[:,entries]) # Doesnt work
    log_abs_det_jacobian = log_abs_det_jacobian[:,entries]


    return torch.sum(log_probs_base + log_abs_det_jacobian, dim=1)


    ll_num   = generalized_conditional_ll(model, x, y1, args)
    ll_denom = generalized_conditional_ll(model, x, y2, args)
    total_woe = ll_num - ll_denom


    nblocks = input_size // k**2
    nrowb = ncolb = int(np.sqrt(nblocks))

    partial_woes = torch.zeros(x.shape[0], nblocks)

    if plot:
        # If no specific index provided We'll display the examlpe with the max total woe
        if exid is None:
            max_woe, exid = total_woe.max(0)
        print(exid)
        if rgba:
            plt.imshow(logistic(x[exid,:]).reshape(28,28))
        else:
            plt.imshow(logistic(x[exid,:]).reshape(28,28), cmap = 'Greys')
        plt.title('Total Woe: {:8.2f}'.format(total_woe[exid]))
        plt.show()

        if rgba:
            cmap = plt.get_cmap()
            full_img = cmap(logistic(x[exid,:]))
            partial_img = 0*full_img
        else:
            full_img = logistic(x[exid,:])
            partial_img = full_img*0 + .1


    #ll_num_prev = ll_denom_prev = torch.zeros_like(x[:,0]) - 10000000000

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
        # if cumulative:
        #     ll_num   = generalized_conditional_ll(model, x, y1, args, block_idxs)
        #     ll_denom = generalized_conditional_ll(model, x, y2, args, block_idxs)
        # else:
        #     ll_num   = (generalized_conditional_ll(model, x, y1, args, block_idxs).exp() - ll_num_prev.exp()).log()
        #     ll_denom = (generalized_conditional_ll(model, x, y2, args, block_idxs).exp() - ll_denom_prev.exp()).log()
        #     ll_num_prev = ll_num
        #     ll_denom_prev = ll_denom


        #ll_num_block =
        woe_block = ll_num - ll_denom
        partial_woes[:, bidx] = woe_block

    if plot:
        plt.figure(figsize = figsize)
        gs1 = gridspec.GridSpec(nrowb, ncolb)
        gs1.update(wspace=0.00, hspace=0.2) # set the spacing between axes.
        woe_ranks = sp.stats.rankdata(-partial_woes[exid,:].detach().numpy(), method = 'ordinal')
        for nb in range(nblocks):
            bidx = block_order_idxs[nb]
            I  = nb // (28//k)
            J  = nb % (28//k)
            init_idx = nb*(k**2)
            block_idxs = input_order_idxs[init_idx:init_idx+k**2]
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
            w, h   = k, k
            rects = patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rects)
            title = '{:2.2f}'.format(partial_woes[exid,bidx])
            if partial_woes[exid,bidx] > 10:
                title = r"$\bf{" + title + "}$"
            ax.set_title('Woe ' + title + ' ({}/{})'.format(woe_ranks[nb], nblocks))

        # Grey out these pixels.
        if rgba:
            partial_img[block_idxs,3] *= 0.5
        else:
            partial_img[block_idxs] *= 0.25

        plt.show()


    partial_woes = partial_woes.detach().numpy()
    total_woe = total_woe.detach().numpy()
    # if order == 'bwd':
    #     partial_woes = np.fliplr(partial_woes)
    # elif order == 'rnd':
    #     partial_woes = [partial_woes[i] for i in block_order_idxs]

    return total_woe, partial_woes #print(partial_woes.sum(axis=1).shape)







def ranking_agreement_score(u, v, k=10):
    """Agreement at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_examples, n_features] or [n_features]
        Scores
    y_score : array-like, shape = [n_examples, n_features] or [n_features]
        Scores
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """

    order_u = np.flip(np.argsort(u, axis = -1), axis = -1)
    order_v = np.flip(np.argsort(v, axis = -1), axis = -1)
#     # Single-dim equivalent:
#     order_u = np.argsort(u)[::-1]
#     order_v = np.argsort(v)[::-1]


    topk_u  = order_u[...,:k] # ... should work for both 1D and 2D
    topk_v  = order_v[...,:k]

    eff_size = float(min(min(len(order_u), len(order_v)), k))

    if topk_u.ndim == 1:
        agreement = np.size(np.intersect1d(topk_u, topk_v))/eff_size
    else:
        agreement = [np.size(np.intersect1d(topk_u[i], topk_v[i]))/eff_size for i in range(topk_u.shape[0])]
    return agreement

def compute_ktau(u, v, verbose = False):
    """
        Computes kendall-tau for ranking.
    """
    def _pairwise_ktau(u, v):
        # U, V are arrays of scalars
        ru = sp.stats.rankdata(u, method = 'ordinal')
        rv = sp.stats.rankdata(v, method = 'ordinal')
        rho, pvalue  = sp.stats.kendalltau(ru,rv)
        if verbose:
            print(u)
            print(ru)
            print(v)
            print(rv)
            print(rho, pvalue)
        return rho, pvalue

    if u.ndim == 1 and v.ndim == 1:
        return _pairwise_ktau(u,v)
    else:
        # This is equivalent to numpy's apply_along_dim but generalized to take two args
        rhos = np.empty_like(u[:,0])
        pvalues = np.empty_like(u[:,0])
        for i,(x,y) in enumerate(zip(u,v)):
            rho, pvalue = _pairwise_ktau(x,y)
            rhos[i] = rho
            pvalues[i] = pvalue
        return rhos, pvalues


def compare_model_woes(model_a, model_b, order_a, order_b, dataloader, h1, h2, args, filter_woes = None):
    total_woes = {'model_a': [], 'model_b': []}
    partial_woes = {'model_a': [], 'model_b': []}
    mse = []
    rhos = []
    pvalues = []
    precisions = {1: [], 3: [], 5: [], 8: []}

    for b, (x,y_true) in enumerate(dataloader):

        total_woe_a, partial_woes_a = decomposition_woe(model_a, x, h1, h2, args, plot = False, order = order_a)

        total_woes['model_a'].append(total_woe_a)
        partial_woes['model_a'].append(partial_woes_a)

        total_woe_b, partial_woes_b = decomposition_woe(model_b, x, h1, h2, args,plot = False, order = order_b)

        total_woes['model_b'].append(total_woe_b)
        partial_woes['model_b'].append(partial_woes_b)

        mse.append(np.linalg.norm(partial_woes_a -  partial_woes_b, axis = 1)**2)

        r, p = compute_ktau(partial_woes_a, partial_woes_b)
        rhos.append(r)
        pvalues.append(p)

        for k,_ in precisions.items():
            precisions[k].append(ranking_agreement_score(partial_woes_a, partial_woes_b, k))

        #pdb.set_trace()
        print('Batch: \t{}. MSE: {:8.2f}. RAS@1: {:2.2f}%. RAS@5: {:2.2f}%.'.format(
            b, mse[-1].mean(), np.array(precisions[1][-1]).mean()*100, np.array(precisions[5][-1]).mean()*100),  end="\r")


    for model_name in ['model_a', 'model_b']:
        total_woes[model_name] = np.concatenate(total_woes[model_name])
        partial_woes[model_name] = np.concatenate(partial_woes[model_name], axis=0)

    for k,v in precisions.items():
        precisions[k] = np.concatenate(v)
        print('Mean agreement at top-{}: {:2.2f}%'.format(k, 100*precisions[k].mean()))


    mse = np.concatenate(mse)
    rhos = np.concatenate(rhos)
    pvalues = np.concatenate(pvalues)

    plt.plot(rhos)
    plt.title('Average corr: {:8.2f}'.format(rhos.mean()))
    plt.show()
    plt.plot(mse)
    plt.title('Average mse: {:8.2f}'.format(mse.mean()))
    plt.show()

    if filter_woes is not None:
        print('Considering only examples where total WOE difference < {}%'.format(100*filter_woes))
        idxskeep = np.abs(total_woes['model_a'] - total_woes['model_b'])/np.abs(total_woes['model_a']) < filter_woes
        for k,v in precisions.items():
            print('Mean agreement at top-{}: {:2.2f}%'.format(k, 100*precisions[k][idxskeep].mean()))
        mse = np.linalg.norm(partial_woes['model_a'][idxskeep,:] -  partial_woes['model_b'][idxskeep,:], axis = 1)**2
        rhos, p = compute_ktau(partial_woes['model_a'][idxskeep,:], partial_woes['model_b'][idxskeep,:])
        plt.plot(rhos)
        plt.title('Average corr: {:8.2f}'.format(rhos.mean()))
        plt.show()
        plt.plot(mse)
        plt.title('Average mse: {:8.2f}'.format(mse.mean()))
        plt.show()
