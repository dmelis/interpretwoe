import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader
from torch.utils.data import TensorDataset


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
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize(size=(32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 1 - x) # To invert B & W
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


def load_mnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = MNIST('data/processed/MNIST', train=True, download=True, transform=transform)
    test  = MNIST('data/processed/MNIST', train=False, download=True, transform=transform)

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

def load_leafsnap_data(data_dir, splits=(0.1,0.1), shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Pad((70,50,0,0)),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.Resize(size=(64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 1 - x) # To invert B & W
    ])

    data = torchvision.datasets.ImageFolder(root=data_dir, transform = transform)

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


    return train_loader, valid_loader, test_loader, data#, sym2idx, idx2sym

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
