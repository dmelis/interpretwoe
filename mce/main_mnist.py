import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator

import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

import matplotlib
matplotlib.use('Agg')

def load_mnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train = MNIST('../../data/MNIST', train=True, download=True, transform=transform)
    test  = MNIST('../../data/MNIST', train=False, download=True, transform=transform)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test

def parse_args():

    #senn_parser = get_senn_parser()
    # [args_senn, extra_args] = senn_parser.parse_known_args()

    #
    ### Local ones
    parser = argparse.ArgumentParser(add_help=False,
        description='Interpteratbility robustness evaluation on MNIST')

  #setup
    #parser.add_argument('--train', action='store_true', default=True, help='Whether or not to train model')
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')

    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--load_model', action='store_true', default=False, help='Load pretrained model from default path')

    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use.')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode' )

    # learning
    parser.add_argument('--optim', type=str, default='adam', help='optim method [default: adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    #parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--epochs_classif', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--epochs_meta', type=int, default=10, help='number of epochs for train [default: 10]')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')

    #paths
    parser.add_argument('--model_path', type=str, default='../checkpoints', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='../out', help='where to dump model config and epoch stats')
    parser.add_argument('--log_path', type=str, default='../log', help='where to dump training logs  epoch stats (and config??)')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch stats')

    # data loading
    parser.add_argument('--num_workers' , type=int, default=4, help='num workers for data loader')


    #####

    args = parser.parse_args()

    # # update args and print
    # args.filters = [int(k) for k in args.filters.split(',')]
    # if args.objective == 'mse':
    #     args.num_class = 1

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


from models import mnist_classifier
from models import masked_mnist_classifier
from utils import generate_dir_names

def main():
    args = parse_args()
    args.nclasses = 10
    model_path, log_path, results_path = generate_dir_names('mnist', args)

    # Load data
    train_loader, valid_loader, test_loader, train_tds, test_tds = load_mnist_data(
                        batch_size=args.batch_size,num_workers=args.num_workers
                        )

    # Train or load classifier
    if args.train or (not os.path.isfile(os.path.join(model_path, "classif.pth"))):
        print('Training classifier from scratch')
        model = mnist_classifier(optim = args.optim, use_cuda = args.cuda)
        model.train(train_loader, test_loader, epochs = args.epochs_classif)
        model.save(os.path.join(model_path, "classif.pth"))
    else:
        print('Loading pre-trained classifier')
        model = torch.load(os.path.join(model_path, "classif.pth"))
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label training examples with mnist_classifier
    # TODO: Before I was reloading dataset with shuffle=False. But it seems this
    # is not necessary, since I assign it to
    train_loader_unshuffled = dataloader.DataLoader(train_tds, shuffle=False, batch_size=args.batch_size)
    train_pred = model.label_datatset(train_loader_unshuffled)
    train_loader.dataset.train_labels = train_pred

    # Train meta model
    mask_size = (7,7)
    padding = 4
    mask_type = 'overlapping'
    mask_model = masked_mnist_classifier(optim = args.optim, mask_type = mask_type,
                                         padding = padding, mask_size = mask_size)
    mask_model.train(train_loader, test_loader, epochs = args.epochs_meta)
    mask_model.save(os.path.join(model_path, "mask_model_{}x{}.pth".format(*mask_size)))
    #


if __name__ == '__main__':
    main()
