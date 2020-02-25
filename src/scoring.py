import numpy as np


def normalized_deltas(hist, pred, alpha = 1, p = 2, reg_type='decay',
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
    if reg_type == 'decay':
        reg = 1/np.power(Cards,p)
    elif reg_type == 'poly-centered':
        argmin = (k+1)/2 if argmin_reg is None else argmin_reg
        #print(argmin)
        maxp   = max(np.abs((1 - argmin))**p, np.abs((len(Cards) - argmin))**p)
        reg = (np.abs(Cards - argmin)**p)/maxp  # Denom normalizes so that maxreg = 1
    else:
        raise ValueError('Unrecognized mode in normalized deltas')
        #reg = torch.pow(torch.abs(Cards - k/2)*(2/k), 3)

    scores = deltas - alpha*reg
    #print(scores)
    # Hack - dont want anything above k/2 card
    # if k > 3:
    #     scores[int(k/2):] = float("-inf")
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


    #m, argm = scores.max(dim=0)  #torc hversion
    argm = scores.argmax()
    m    = scores.max()

    C = inds[:argm+1]
    if len(C) == 0:
        pdb.set_trace()
    #print(asd.asd)
    # Map back to original indices of classes
    #C_orig = [classes[i] for i in C] if subs_k < true_k else C
    #else:
    #print('Selected classes (fake index)', C)
    #print(k, scores)
    return m, C
