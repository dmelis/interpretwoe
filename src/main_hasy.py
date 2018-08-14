import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
from tqdm import tqdm

import matplotlib
if matplotlib.get_backend() == 'Qt5Agg':
    # Means this is being run in server, need to modify backend
    matplotlib.use('Agg')


import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

import torchvision
from torchvision import transforms

### Local Imports
from models import image_classifier, masked_image_classifier
from utils import generate_dir_names
from utils import SubsetDeterministicSampler
#from torchvision import datasets

def read_symbol_dict(fpath):
    sym2ind = {}
    ind2sym = {}
    with open(fpath) as f:
        header = f.readline()
        for line in f:
            splitted = line.split(',')
            i, s = splitted[:2]
            #print(i,s)
            sym2ind[s]=i
            ind2sym[i]=s
    return sym2ind, ind2sym

def load_hasy_data(data_dir, splits=(0.1,0.1), shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    # transform = transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize(size=(32, 32)),
        torchvision.transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x) # To invert B & W
    ])


    data = torchvision.datasets.ImageFolder(root=data_dir, transform = transform)
    #test  = torchvision.datasets.ImageFolder(root=data_dir, transform = transform)

    num_examples = len(data)
    indices = list(range(num_examples))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    valid_size, test_size = splits
    tst_split = int(np.floor(test_size * num_examples))
    val_split = int(np.floor((test_size + valid_size) * num_examples))

    train_idx, valid_idx, test_idx = \
        indices[val_split:], indices[tst_split:val_split], indices[:tst_split]

    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_examples

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler  = SubsetRandomSampler(test_idx)

    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = dataloader.DataLoader(data, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(data, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(data, sampler=test_sampler, **dataloader_args)

    # Get symbol dict
    # This is symbol string to symbol id withing Hasy
    sym2symid, symid2sym = read_symbol_dict(os.path.join(data_dir,'symbols.csv'))
    # Need also to find mapping to the (ordered) set of indices used by torch loader
    sym2idx, idx2sym = {}, []
    for idx in range(len(sym2symid.keys())):
        symid = data.classes[idx]
        sym   = symid2sym[symid]
        idx2sym.append(sym)
        sym2idx[sym] = idx

    return train_loader, valid_loader, test_loader, data, sym2idx, idx2sym


def load_full_dataset(loader, to_numpy = False, max_examples = None):
    print('Loading and stacking image data....')
    X_full, Y_full = [], []
    tot = 0
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        #print(batch_idx)
        X_full.append(data)
        Y_full.append(target)
        tot += data.shape[0]
        # if max_examples and (tot > max_examples):
        #     break
    if to_numpy:
        X_full = torch.cat(X_full).numpy().astype('float32')
        Y_full = torch.cat(Y_full).numpy()
    else:
        X_full = torch.cat(X_full)
        Y_full = torch.cat(Y_full)
    if max_examples:
        idxs = torch.randperm(X_full.shape[0])[:max_examples]
        print(idxs[:10])
        X_full = X_full[idxs]
        Y_full = Y_full[idxs]
    return X_full, Y_full

def parse_args():

    #senn_parser = get_senn_parser()
    # [args_senn, extra_args] = senn_parser.parse_known_args()

    #
    ### Local ones
    parser = argparse.ArgumentParser(add_help=False,
        description='Interpteratbility robustness evaluation on HASY dataset')

    parser.add_argument('--train-classif', action='store_true', default=False, help='Whether or not to (re)train classifier model')
    parser.add_argument('--train-meta', action='store_true', default=False, help='Whether or not to (re)train meta masking model')

    # Meta-learner
    parser.add_argument('--attrib_type', type=str, choices = ['overlapping'],
                            default='overlapping', help='type of attribute')
    parser.add_argument('--attrib_width', type=int, default=7, help='width of attribute region [default: 7]')
    parser.add_argument('--attrib_height', type=int, default=7, help='height of attribute region [default: 7]')
    parser.add_argument('--attrib_padding', type=int, default=4, help='Padding around input image to define attributes')

    # Explainer
    parser.add_argument('--mce_reg_type', type=str, choices = ['exp', 'quadratic'],
                            default='exp', help='Type of regularization for explainer scoring function objective')
    parser.add_argument('--mce_alpha', type=float,
                            default=1.0, help='Alpha parameter for explainer scoring function')
    parser.add_argument('--mce_p', type=int, default=2, help='P power for explainer scoring function')
    parser.add_argument('--mce_plottype', type=str, choices = ['treemap', 'bar'],
                            default='treemap', help='Plot type of class entailment diagram')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode' )

    # learning
    parser.add_argument('--optim', type=str, default='adam', help='optim method [default: adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs_classif', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--epochs_meta', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')

    #paths
    parser.add_argument('--model_path', type=str, default='../checkpoints', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='../out', help='where to dump model config and epoch stats')
    parser.add_argument('--log_path', type=str, default='../log', help='where to dump training logs  epoch stats (and config??)')

    # data loading
    parser.add_argument('--num_workers' , type=int, default=4, help='num workers for data loader')
    #
    #
    # # device
    # parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    # parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use.')
    # parser.add_argument('--debug', action='store_true', default=False, help='debug mode' )
    #
    # # learning
    # parser.add_argument('--optim', type=str, default='adam', help='optim method [default: adam]')
    # parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    # #parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    # parser.add_argument('--epochs_classif', type=int, default=10, help='number of epochs for train [default: 10]')
    # parser.add_argument('--epochs_meta', type=int, default=10, help='number of epochs for train [default: 10]')
    #
    # parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    # parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')
    #
    # #paths
    # parser.add_argument('--model_path', type=str, default='../checkpoints', help='where to save the snapshot')
    # parser.add_argument('--results_path', type=str, default='../out', help='where to dump model config and epoch stats')
    # parser.add_argument('--log_path', type=str, default='../log', help='where to dump training logs  epoch stats (and config??)')
    # parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch stats')
    #
    # # data loading
    # parser.add_argument('--num_workers' , type=int, default=1, help='num workers for data loader')
    # #####

    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def main():
    args = parse_args()
    model_path, log_path, results_path = generate_dir_names('hasy', args)

    ### LOAD DATA
    print('Loading data...', end='')
    train_loader, valid_loader, test_loader, dataset, sym2ind, ind2sym  = \
        load_hasy_data('../../data/Hasy/', splits = (0.1,0.1),
        batch_size=args.batch_size,num_workers=args.num_workers)
    print('done!')
    classes = dataset.classes
    args.nclasses = len(dataset.classes)

    ### TRAIN OR LOAD CLASSIFICATION MODEL TO BE EXPLAINED
    classif_path = os.path.join(model_path, "classif.pth")
    if args.train_classif or (not os.path.isfile(classif_path)):
        print('Training classifier from scratch')
        clf = image_classifier(task = 'hasy', optim = args.optim, use_cuda = args.cuda)
        clf.train(train_loader, valid_loader, epochs = args.epochs_classif)
        clf.save(classif_path)
    else:
        print('Loading pre-trained classifier')
        clf = image_classifier.load(classif_path)
        clf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.test(test_loader)

    # Label all examples with the classifier (have no easy way to labeling only train,
    # but it seems it doesn't matter, labels for others wont' be used anyway)
    train_idx = sorted(train_loader.sampler.indices)
    dataloader_args = dict(batch_size=1000, num_workers=0, shuffle = False)
    dummy_loader = dataloader.DataLoader(dataset,
                sampler=SubsetDeterministicSampler(train_idx), **dataloader_args)

    # Get reference examples for nearest neighbor computation in masked classifier
    X_full, Y_full = load_full_dataset(dummy_loader, to_numpy=True, max_examples = 20000)

    ### TRAIN OR LOAD META-MODEL OF MASKED INPUTS
    mask_size = (args.attrib_width, args.attrib_height)
    metam_path = os.path.join(model_path, "mask_model_{}x{}.pth".format(*mask_size))
    if args.train_meta or (not os.path.isfile(metam_path)):
        print('Training meta masking model from scratch')
        mask_model = masked_image_classifier(task = 'hasy', X = X_full, Y = Y_full,
                                             optim = args.optim,
                                             mask_type = args.attrib_type,
                                             padding = args.attrib_padding,
                                             mask_size = mask_size)
        mask_model.train(train_loader, test_loader, epochs = args.epochs_meta)
        mask_model.save(metam_path)
    else:
        print('Loading pre-trained meta masking model')
        mask_model = masked_image_classifier.load(metam_path)
        mask_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### EXPLAIN SOME INSTANCES
    Explainer = MCExplainer(clf, mask_model, classes = classes,
                      reg_type = args.mce_reg_type,
                      crit_alpha = args.mce_alpha,
                      crit_p = args.mce_p,
                      plot_type = args.mce_plottype)

    # Grab a batch for experiments
    batch_x, batch_y = next(iter(train_loader))
    idx = 63
    x  = batch_x[idx:idx+1]
    print(classes[batch_y[idx].item()])

    e = Explainer.explain(x, verbose = 0 , show_plot = 1)

if __name__ == '__main__':
    main()