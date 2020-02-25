
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



################################################################################
