import os
import numpy as np
import math
import itertools
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches
import matplotlib.gridspec as gridspec

import pdb
from torchvision.utils import make_grid

try:
    from IPython.display import clear_output
except ImportError:
    pass # or set "debug" to something else or whatever

import shapely
import shapely.geometry #import MultiPolygon
import shapely.ops
import matplotlib
import squarify
from torch.utils.data.sampler import Sampler


def tensorshow(img, greys = False):
    """
        For images in pytorch tensor format
    """
    npimg = img.numpy()
    if greys:
        plt.imshow(npimg[0], interpolation='nearest', cmap = 'Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()



class SubsetDeterministicSampler(Sampler):
    r"""Samples elements sequentially (deterministically) from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


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
    """
        Should deprecate in favor of more general image_masker below
    """
    ii = list(range(H)[S[0]])[0]
    jj = list(range(W)[S[1]])[0]
    mask = torch.zeros(H, W) # FIXME: SHOULD BE TRUE ZERO
    mask[S[0],S[1]] = 1
    ### unnormalize should be abstracted away - be part of woe_model and classif respectively?
    ### or maybe of dataset class
    X_s = mnist_unnormalize(x)*mask.view(1,1,H,W).repeat(x.shape[0], 1, 1, 1)
    X_plot  = X_s
    X_input = mnist_normalize(X_s)
    return X_input, X_plot


def image_masker(x, S, H = 32, W = 32, normalized = False, alpha = 0):
    """
        Should work for other tasks where we don't prenormalize. Considering renaming
        to something more general.

        Since input doesn't require normalization, X_plot and X_input are the same.
    """
    ii = list(range(H)[S[0]])[0]
    jj = list(range(W)[S[1]])[0]
    mask = torch.zeros(H, W) + alpha# FIXME: SHOULD BE TRUE ZERO
    mask[S[0],S[1]] = 1
    if normalized:
        X_s = mnist_unnormalize(x)*mask.view(1,1,H,W).repeat(x.shape[0], 1, 1, 1)
        X_plot  = X_s
        X_input = mnist_normalize(X_s)
    else:
        X_s = x*mask.view(1,1,H,W).repeat(x.shape[0], 1, 1, 1)
    return X_s, X_s


################################################################################
