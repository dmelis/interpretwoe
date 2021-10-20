import os, sys
from attrdict import AttrDict
import argparse
import numpy as np
import scipy as sp
try:
    import torch
except:
    print('no pytorch')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

DEBUG = False


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
