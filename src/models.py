import os
import sys
import pdb
from tqdm import tqdm
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches


import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, WeightedRandomSampler
import torch.nn as nn

# Torchnet (tnt) utils
from torchnet.meter.confusionmeter import ConfusionMeter


try:
    sys.path.append('/home/t-daalv/pkg/faiss/python')
    import faiss # Only needed by mask mnist for fast knn search
except:
    print("FAISS library not found - will note be able to retrain masking models")



# Local Imports

from .utils import plot_confusion_matrix


#===============================================================================
#=================================  NETWORKS ===================================
#===============================================================================

def mnist_unnormalize(x):
    return x.clone().mul_(.3081).add_(.1307)

def mnist_normalize(x):
    return x.clone().sub_(.1307).div_(.3081)

class MnistNet(nn.Module):
    def __init__(self, final_nonlin = 'log_sm', nclasses = 10):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)
        self.final_nonlin = final_nonlin

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.final_nonlin == 'log_sm':
            x = F.log_softmax(x, dim = 1)
        elif self.final_nonlin == 'sigmoid':
            #x = F.sigmoid(x)
            x = torch.sigmoid(x)
        return x

class LeafNet(nn.Module):
    """
        CNN for Leafsnap dataset.
    """
    def __init__(self, final_nonlin = 'log_sm', nclasses = 185):
        super(LeafNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*14*14, 1024)
        self.fc2 = nn.Linear(1024, nclasses)
        self.final_nonlin = final_nonlin

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride = 2))   # 128 x 32 x 31 x31
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, stride = 2)) # 128 x 64 x 14 x 14
        x = x.view(-1, 64*14*14)
        #pdb.set_trace()
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.final_nonlin == 'log_sm':
            x = F.log_softmax(x, dim = 1)
        elif self.final_nonlin == 'sigmoid':
            x = torch.sigmoid(x)
        return x

# 128x128 model
# class LeafNet(nn.Module):
#     """
#         CNN for Leafsnap dataset.
#     """
#     def __init__(self, final_nonlin = 'log_sm', nclasses = 185):
#         super(LeafNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(64*30*30, 1024)
#         self.fc2 = nn.Linear(1024, nclasses)
#         self.final_nonlin = final_nonlin
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2, stride = 2)) # 128 x 32 x 63 x 63
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, stride = 2)) # 128 x 64 x 30 x 30
#         x = x.view(-1, 64*30*30)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         if self.final_nonlin == 'log_sm':
#             x = F.log_softmax(x, dim = 1)
#         elif self.final_nonlin == 'sigmoid':
#             x = torch.sigmoid(x)
#         return x

class HasyNet(nn.Module):
    def __init__(self, final_nonlin = 'log_sm', nclasses = 369):
        super(HasyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*6*6, 1024)
        self.fc2 = nn.Linear(1024, nclasses)
        self.final_nonlin = final_nonlin

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, stride = 2))
        x = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.final_nonlin == 'log_sm':
            x = F.log_softmax(x, dim = 1)
        elif self.final_nonlin == 'sigmoid':
            x = torch.sigmoid(x)
        return x

class NGramCNN(nn.Module):
    def __init__(self, ngram = 5, hidden_dim = 300, emb_dim = None, vocab = None, vocab_size = None,
                 final_nonlin = 'log_sm', nclasses = 369):
        super(NGramCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nclasses   = nclasses
        self.vocab = vocab
        padding_idx = vocab.stoi['<PAD>']
        if vocab.vectors is not None:
            print('Using pretrained word embeddings')
            self.emb_dim = vocab.vectors.shape[1]
            self.vocab_size = vocab.vectors.shape[0]
            self.embed = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx)
            self.embed.weight.data.copy_(vocab.vectors)
        else:
            self.emb_dim = emb_dim
            self.embed = nn.Embedding(vocab_size, self.emb_dim, padding_idx)

        self.final_nonlin = final_nonlin

        self.hidden_dim = hidden_dim
        Ci = 1       # Input channels
        Co = 100     # Number of output channels for each filter
        Ks = [2,3,min(ngram, 4)] # Kernel sizes
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, self.emb_dim)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        #self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, 32)
        self.fc2 = nn.Linear(32, nclasses)


    def forward(self, x):
        #pdb.set_trace()
        x = self.embed(x)      # (B, N, EmbD)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = F.dropout(x, training=self.training) # (N, len(Ks)*Co)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.final_nonlin == 'log_sm':
            x = F.log_softmax(x, dim = 1)
        elif self.final_nonlin == 'sigmoid':
            #x = F.sigmoid(x)
            x = torch.sigmoid(x)
        return x


# Make AE Model
class FFCAE(nn.Module):
    def __init__(self):
        super(FFCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.ReLU(True),
            nn.Linear(28 * 28, 28 * 28),
            nn.Hardtanh(min_val = -0.4242, max_val = 2.8215)
            )

    def forward(self, x):
        self.e = self.encoder(x.view(-1,28*28))
        return self.decoder(self.e).view(x.shape)

    def samples_write(self, x, epoch):
        # Writing data in a grid to check the quality and progress
        samples = self.forward(x)
        samples = samples.data.cpu().numpy()[:16]
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        if not os.path.exists('out/'):
            os.makedirs('out/')
        plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Hardtanh(min_val = -0.4242, max_val = 2.8215)
        )

    def forward(self, x):
        self.e = self.encoder(x)#.view(-1,28*28))
        return self.decoder(self.e).view(x.shape)

    def samples_write(self, x, epoch):
        samples = self.forward(x)
        samples = samples.data.cpu().numpy()[:16]
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        if not os.path.exists('out/'):
            os.makedirs('out/')
        plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')


#===============================================================================
#======================  MODEL WRAPPERS FOR TRAINING  ==========================
#===============================================================================


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, vocab,
                num_layers = 2, dropout = 0.5, device = 'cpu'):
        super(LSTMClassifier, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device_str = device
        self.hidden_dim = hidden_dim
        self.nclasses   = tagset_size

        self.vocab = vocab
        padding_idx = vocab.stoi['<PAD>']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first = True,
                               num_layers = num_layers, dropout = dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, inputs, lengths=None):
        """
            inputs: (seqlen, batch)
        """
        batch_size, seqlen = inputs.size()

        # 1. Embedding Layer
        #  (batch, seqlen) ->  (batch, seqlen, embdim)
        embeds = self.word_embeddings(inputs)

        # 2. Run through rnn
        # (batch, seqlen, embdim) -> (batch, seqlen, hiddendim)
        packed_emb = embeds
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first = True)
        output, self.hidden = self.encoder(packed_emb)  # embed_input

        # Undo unpacking
        if lengths is not None:
             output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first = True)

        # 3. Get last state
        time_dimension = 1 # since time_dim = (batch_first == 1)
        idx = (torch.LongTensor(lengths).to(torch.device(self.device_str)) - 1).view(-1, 1).expand(
            len(lengths), output.size(2))
        idx = idx.unsqueeze(time_dimension) # 1 is time dimension
        #print('idx', idx.shape)

        last_output = output.gather(time_dimension, idx).squeeze(time_dimension)

        #print('last', last_output.shape)
        # 4. Project to label space

        # (batch, seqlen, hiddendim) -> (batch*seqlen, hiddendim)
        #output = output.contiguous().view(-1, output.shape[2])

        # (batch*seqlen, hiddendim) -> (batch*seqlen, nclasses)
        tag_space = self.hidden2tag(last_output) #len(sentence), -1))

        # (batch*seqlen, nclasses) -> (batch, seqlen, nclasses)
        tag_scores =  F.log_softmax(tag_space, dim=1).view(batch_size, self.nclasses)

        return tag_scores


class text_classifier(nn.Module):
    def __init__(self, vocab, langs, optim = 'adam', log_interval = 10, use_cuda = False,
        hidden_dim = 100, dropout = 0.5, weight_decay = 1e-06, lr = 0.001,
        num_layers = 1, **kwarg):
        super(ets_classifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = log_interval
        self.emb_dim = vocab.vectors.shape[1]
        self.hidden_dim = hidden_dim
        self.weight_decay = weight_decay
        self.vocab = vocab
        self.langs = langs
        self.dropout = dropout
        self.num_layers = num_layers
        self.lr = lr
        # Need to pass vocab, etc to LSTM classifier
        self.net = LSTMClassifier(self.emb_dim, hidden_dim, len(vocab),
                len(langs), vocab = vocab, device = self.device,
                num_layers = num_layers, dropout=dropout).to(self.device)
        if vocab.vectors is not None:
            print('Using pretrained word embeddings')
            self.net.word_embeddings.weight.data.copy_(vocab.vectors)

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=lr, weight_decay=weight_decay)

    def __call__(self, x, lengths = None):
        return self.net(x, lengths)

    # @staticmethod
    # def load(path):
    #     if 'gpu' in path:
    #         model = torch.load(path,
    #                     map_location=lambda storage, location: storage)
    #     else:
    #         model = torch.load(path)
    #     dev = "cuda" if torch.cuda.is_available() else "cpu"
    #     model.device = torch.device(dev)
    #     model.net.device_str = dev
    #     return model

    @staticmethod
    def load(path):
        model =  torch.load(path,
                        map_location=lambda storage, location: storage)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.device = torch.device(dev)
        model.net.device_str = dev
        return model

    def save(self, path):
        # Hack - serialization of torch.device is very recent https://github.com/pytorch/pytorch/pull/7713
        device = self.device
        self.device = None
        torch.save(self, open(path, 'wb'))
        self.device = device
        print('Saved!')

    def train(self, train_loader, test_loader, epochs = 2):
        # Train classifier
        #use_cuda = False
        #device = torch.device("cuda" if use_cuda else "cpu")
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader, plot = True)

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        for batch_idx, batch in enumerate(train_loader):
            (inputs, lengths), targets = batch.text, batch.label
            inputs, lengths, targets = inputs.to(self.device), lengths.to(self.device), targets.to(self.device)

            #print(inputs.shape)
            #(x, x_lengths), y = data.text, data.label
            self.net.zero_grad()
            self.net.hidden = self.net.init_hidden()

            pred_scores = self.net(inputs, lengths)
            #print(pred_scores.shape)
            loss = F.nll_loss(pred_scores, targets)

            loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, test_loader, plot = True, save_path = None):
        self.net.eval()
        test_loss = 0
        correct = 0
        conf = ConfusionMeter(len(self.langs))
        with torch.no_grad():
            for batch in tqdm(test_loader):
                (inputs, lengths), targets = batch.text, batch.label
                inputs, lengths, targets = inputs.to(self.device), lengths.to(self.device), targets.to(self.device)
                self.net.hidden = self.net.init_hidden()
                output = self.net(inputs, lengths)
                test_loss += F.nll_loss(output, targets, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()
                conf.add(pred.squeeze(), targets.squeeze())

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        if plot:
            plot_confusion_matrix(conf.value(), self.langs)
            plt.show()
        else:
            print(conf.value())

    def label_datatset(self, loader):
        print('Labeling dataset...')
        self.net.eval()
        fx = []
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(tqdm(loader)):
                (inputs, lengths), targets = batch.text, batch.label
                inputs, lengths, targets = inputs.to(self.device), lengths.to(self.device), targets.to(self.device)
                self.net.hidden = self.net.init_hidden()
                out = self.net(inputs, lengths)
                p, pred = out.max(1)
                fx.append(pred.cpu())
        return torch.cat(fx)

class image_classifier(nn.Module):
    def __init__(self, task = 'mnist', optim = 'sgd', log_interval = 10, use_cuda = False, **kwarg):
        super(image_classifier, self).__init__()
        self.device_type = "gpu" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = log_interval
        if task == 'mnist':
            self.net = MnistNet().to(self.device)
        elif task == 'hasy':
            self.net = HasyNet().to(self.device)
        elif task == 'leafsnap':
            self.net = LeafNet().to(self.device)
        else:
            raise ValueError("Unrecoginzed task in image_classifier")

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)

    def __call__(self, x):
        return self.net(x)

    def save(self, path):
        # Hack - serialization of torch.device is very recent https://github.com/pytorch/pytorch/pull/7713
        device = self.device
        self.device = None
        fname = path #+ '.gpu' if self.device_type == 'gpu' else '' + '.pth'
        torch.save(self, open(fname, 'wb'))
        self.device = device
        print('Saved!')

    @staticmethod
    def load(path):
        model =  torch.load(path,
                        map_location=lambda storage, location: storage)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.device = torch.device(dev)
        return model

    def train(self, train_loader, test_loader, epochs = 2):
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader)

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, test_loader):
        self.net.eval()
        test_loss = 0
        correct = 0
        # len(test_loader.dataset is wrong if using subsetsampler!)
        if type(test_loader.sampler) is SequentialSampler:
            ntest = len(test_loader.dataset)
        elif type(test_loader.sampler) is SubsetRandomSampler:
            ntest = len(test_loader.sampler.indices)
        else:
            pdb.set_trace()
            raise ValueError("Wrong sampler")

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= ntest
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, ntest, 100. * correct / ntest))

    def label_datatset(self, loader):
        print('Labeling dataset...')
        self.net.eval()
        fx = []
        for idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(self.device)
            out = self.net(data)
            p, pred = out.max(1)
            fx.append(pred.cpu())
        return torch.cat(fx)


class mnist_autoencoder():
    def __init__(self, optim, use_cuda = False, *args, **kwargs):
        super(mnist_autoencoder, self).__init__()
        #self.net = FFCAE()
        self.net = ConvAE()
        self.log_interval = kwargs['log_interval']
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)


    def __call__(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net, open(path, 'wb'))
        print('Saved!')

    def train(self, train_loader, test_loader, epochs = 2):
        # Train classifier
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader)

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        for batch_idx, (data,_) in enumerate(train_loader):
            data, target = data.to(self.device), data.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, test_loader):
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data, target = data.to(self.device), data.to(self.device)
                output = self.net(data)
                test_loss += F.mse_loss(output, target, size_average=True).item() # sum up batch loss

        test_loss /= len(test_loader.dataset)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data[0].squeeze())
        ax[1].imshow(output[0].squeeze())
        plt.show()

        print('\nTest set: Average loss: {:.8f}\n'.format(test_loss))

### TOOLS FOR CONTRACTION TRAINERS

def jacobian(inputs, outputs):
    return torch.stack([torch.autograd.grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=True)[0]
                        for i in range(outputs.size(1))], dim=-1)


class contractive_ae():
    def __init__(self, optim = 'adam', use_cuda = False, nclass = 10, l_contr = 1,
        l_anchor = 0, eps_c = 0.2, anchor = None, denoising = False,  *args, **kwargs):
        super(contractive_ae, self).__init__()
        self.net = FFCAE()
        self.log_interval = kwargs['log_interval']
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.nclass = nclass
        self.batch_size = kwargs['batch_size']
        self.lambda_contraction = l_contr
        self.lambda_anchor      = l_anchor
        self.eps_contraction = eps_c
        self.denoising       = denoising
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-5)
        if anchor is not None:
            self.x_anchor = anchor
            self.use_anchor_loss = True
        else:
            self.use_anchor_loss = False

    def __call__(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net, open(path, 'wb'))
        print('Saved!')

    def train(self,  train_loader, test_loader, epochs = 2):
        # Train classifier
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader, epoch)

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        train_loss = 0
        if self.denoising:
            noise = torch.rand(self.batch_size,1,28,28)
        for idx, (data,_) in enumerate(train_loader):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), data.to(self.device).detach()
            if self.denoising:
                if data.shape[0] != noise.shape[0]: # last batch
                    noise = torch.rand(data.shape[0],1,28,28)
                data = torch.mul(data+0.25, 0.1 * noise)
            data.requires_grad = True

            # 1. Reconstruction loss
            recons_x = self.net(data)
            recons_loss = F.mse_loss(recons_x, target) #+ lam*grad_penalty
            losses = [recons_loss]

            # 2. Contraction Loss - via Jacobian
            dF = jacobian(data, recons_x)
            #grad_penalty = (dF).norm(2) #.pow(2) # ABSOLUTE
            contraction_loss = (dF.norm(2) - (1 - self.eps_contraction)).clamp(min = 0) # Only penalize above eps
            losses.append(self.lambda_contraction*contraction_loss)

            # 3. Optional: Anchor loss to guide fixed point
            if self.use_anchor_loss:
                anchor_loss = F.mse_loss(self.x_anchor,self.net(self.x_anchor))
                losses.append(self.lambda_anchor*anchor_loss)
            else:
                anchor_loss = torch.zeros(1)

            sum(losses).backward()
            self.optimizer.step()

            train_loss += sum(losses).item()

            if idx % self.log_interval == 0:
                print('Train epoch: {} [{}/{}({:.0f}%)]\t '
                 'Rec. Loss: {:.2e}\t Contr. Loss: {:.2e}\t FP Loss: {:.2e}'.format(
                  epoch, idx*len(data), len(train_loader.dataset),
                  100*idx/len(train_loader),
                  recons_loss.item()/len(data), contraction_loss.item()/len(data),
                  anchor_loss.item()/len(data)))

        print('====> Epoch: {} Average training loss: {:.8f}'.format(
             epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader, epoch):
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data, target = data.to(self.device), data.to(self.device)
                output = self.net(data)
                batch_loss = F.mse_loss(output, target, size_average=True).item()
                #batch_loss += (dF.norm(2) - (1 - self.eps_contraction)).clamp(min = 0)
                test_loss +=  batch_loss

                #TODO: add contractive loss here too
        self.net.samples_write(data,epoch)
        plt.show()
        test_loss /= len(test_loader.dataset)
        print('====> Epoch: {} Average training loss: {:.8f}\n\n'.format(epoch, test_loss))

        if self.use_anchor_loss:
            x_star_rec =self.net(self.x_anchor)
            print((x_star_rec - self.x_anchor).norm().detach().numpy())
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(self.x_anchor.squeeze())
            ax[1].imshow(x_star_rec.detach().squeeze())
            plt.show()




class class_contractive_ae():
    def __init__(self, optim = 'adam', use_cuda = False, nclass = 10, lambd = 1,
        contr_steps = 1, eps_c = 0.2, eps_e = 0.5, *args, **kwargs):
        super(class_contractive_ae, self).__init__()
        self.net = FFCAE()
        self.log_interval = kwargs['log_interval']
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.nclass = nclass
        self.batch_size = kwargs['batch_size']
        self.lambda_contraction = lambd
        self.eps_contraction = eps_c
        self.eps_expansion   = eps_e
        self.steps_contraction = contr_steps
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-5)
        self.class_loaders = {}

    def __call__(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net, open(path, 'wb'))
        print('Saved!')

    def train(self, pred_y, train_loader, test_loader, epochs = 2):
        self.init_class_loaders(train_loader.dataset, pred_y)
        # Train classifier
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            #self.test(test_loader)

    def init_class_loaders(self, dataset, pred_y):
        for klass in range(self.nclass):
            weights = (pred_y == klass)
            sampler = WeightedRandomSampler(weights, len(weights), replacement = True)
            self.class_loaders[klass] = iter(torch.utils.data.DataLoader(dataset=dataset,
                           batch_size=self.batch_size,drop_last = True,
                           sampler = sampler))#, num_workers=args.workers, pin_memory=True)
        #return class_loaders
    def reset_class_loader(self, k):
        sampler = self.class_loaders[k].batch_sampler.sampler
        loader  = torch.utils.data.DataLoader(dataset=self.class_loaders[k].dataset,
                               batch_size=self.batch_size,drop_last = True,
                               sampler = sampler)
        self.class_loaders[k] = iter(loader)#, num_worker

    def contraction_loss(self, XC1, XC2, XC1_r, XC2_r):
        """
            *_r are the outputs (reconstructed)
        """

        # Split into two
        b = self.batch_size
        n = b/2
        XC1_A, XC1_B = torch.split(XC1.view(b,-1), [n, n], dim = 0)
        XC2_A, XC2_B = torch.split(XC2.view(b,-1), [n, n], dim = 0)
        XC1_A_r, XC1_B_r = torch.split(XC1_r.view(b,-1), [n, n], dim = 0)
        XC2_A_r, XC2_B_r = torch.split(XC2_r.view(b,-1), [n, n], dim = 0)

        # Compute Lipschitz Ratios
        # lip_C1  = (XC1_A_r - XC1_B_r).norm()/(XC1_A - XC1_B).norm()
        # lip_C2  = (XC2_A_r - XC2_B_r).norm()/(XC2_A - XC2_B).norm()
        # lip_12  = (XC1_A_r - XC2_A_r).norm()/(XC1_A - XC2_A).norm()
        # lip_21  = (XC1_B_r - XC2_B_r).norm()/(XC1_B - XC2_B).norm()
        # Each of this is (n x 1)
        lip_C1  = (XC1_A_r - XC1_B_r).norm(dim=1)/(XC1_A - XC1_B).norm(dim=1)
        lip_C2  = (XC2_A_r - XC2_B_r).norm(dim=1)/(XC2_A - XC2_B).norm(dim=1)
        lip_12  = (XC1_A_r - XC2_A_r).norm(dim=1)/(XC1_A - XC2_A).norm(dim=1)
        lip_21  = (XC1_B_r - XC2_B_r).norm(dim=1)/(XC1_B - XC2_B).norm(dim=1)


        # For same class, we want contraction: lip <= (1 - eps_c)
        contraction_loss = (lip_C1 -  (1 - self.eps_contraction)).clamp(min = 0) + \
                           (lip_C2 -  (1 - self.eps_contraction)).clamp(min = 0)
        # Across classes, we want expansion:   lip >= (1 + eps_e)
        expansion_loss =  -(lip_12 - (1 + self.eps_expansion)).clamp(max = 0) + \
                          -(lip_21 - (1 + self.eps_expansion)).clamp(max = 0)
        # Total loss is sum of the two
        return contraction_loss.mean() + expansion_loss.mean()

    def _draw_from_class(self, k):
        try:
            x, _ = next(self.class_loaders[k])
            assert x.shape[0] == self.batch_size#class_loaders[klass].batch_sampler.batch_size
        except:
            print('Reset iterator for class: ', k)
            self.reset_class_loader(k)
            # class_loaders[klass] = iter(torch.utils.data.DataLoader(dataset=dataset,
            #        batch_size=args.batch_size,
            #        sampler = sampler))#, num_workers=args.workers, pin_memory=True)
            x, _ = next(self.class_loaders[k])
        return x

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        train_loss = 0
        for idx, (data,_) in enumerate(train_loader):
            data, target = data.to(self.device), data.to(self.device)
            self.optimizer.zero_grad()

            recons_x = self.net(data)
            recon_loss = F.mse_loss(recons_x, data.detach()) #+ lam*grad_penalty
            losses = [recon_loss]

            # Randomly choose a pair of distinct classes, map them
            for i in range(self.steps_contraction):
                c1, c2 = np.random.choice(self.nclass, 2, replace=False)
                XC1 = self._draw_from_class(c1)
                XC2 = self._draw_from_class(c2)
                XC1_rec = self.net(XC1)
                XC2_rec = self.net(XC2)
                contraction_loss = self.contraction_loss(XC1, XC2, XC1_rec, XC2_rec)
                losses.append(self.lambda_contraction*contraction_loss)

            sum(losses).backward()
            self.optimizer.step()
            train_loss += sum(losses).item()

            if idx % self.log_interval == 0:
                print('Train epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.8f}\t '
                 'Rec. Loss: {:.8f}\t Contr. Loss: {:.8f}'.format(
                  epoch, idx*len(data), len(train_loader.dataset),
                  100*idx/len(train_loader),sum(losses).item()/len(data),
                  recon_loss.item()/len(data), contraction_loss.item()/len(data)))

        print('====> Epoch: {} Average loss: {:.8f}'.format(
             epoch, train_loss / len(train_loader.dataset)))
        self.net.samples_write(data,epoch)

    def test(self, test_loader):
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data, target = data.to(self.device), data.to(self.device)
                output = self.net(data)
                test_loss += F.mse_loss(output, target, size_average=True).item() # sum up batch loss

                #TODO: add contractive loss here too

        test_loss /= len(test_loader.dataset)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data[0].squeeze())
        ax[1].imshow(output[0].squeeze())
        plt.show()

        print('\nTest set: Average loss: {:.8f}\n'.format(test_loss))

class masked_image_classifier():
    def __init__(self, task = 'mnist', optim = 'adam', mask_type = 'disjoint',
                missing_method = 'zero', padding = 0, knn = 20, image_size = (28,28),
                mask_size = (14,14), log_interval = 10, X = None, Y = None, **kwarg):
        super(masked_image_classifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = log_interval
        self.plot_interval = 1000
        self.padding = padding
        self.knn = knn
        self.missing_method = 'zero'
        self.mask_size = mask_size
        self.task = task
        self.X, self.Y = X, Y
        if task == 'mnist':
            self.input_size = (28,28)
            self.net = MnistNet(final_nonlin='sigmoid').to(self.device)
            self.nclasses = 10
        elif task == 'hasy':
            self.input_size = (32,32)
            self.net = HasyNet(final_nonlin='sigmoid').to(self.device)
            self.nclasses = 369
        elif task == 'leafsnap':
            self.input_size = (64, 64)
            self.net = LeafNet(final_nonlin='sigmoid').to(self.device)
            self.nclasses = 185
        else:
            raise ValueError("Unrecoginzed task in image_classifier")

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)

        ## Initialize mask corner points
        # I, J contain valid left and top init pixels
        if mask_type == 'disjoint':
            # Partition input into equallty sized, disjoint squares
            self.I = list(range(0, self.input_size[0], mask_size[0]))
            self.J = list(range(0, self.input_size[1], mask_size[1]))
        elif mask_type == 'overlapping':
            # Overlapping squares
            self.I = list(range(self.padding, self.input_size[0] - self.mask_size[0]- self.padding))
            self.J = list(range(self.padding, self.input_size[1] - self.mask_size[1] - self.padding))
        #print('I,J:', self.I, self.J)

    def __call__(self, x):
        return self.net(x)

    def eval(self):
        self.net.eval()

    def save(self, path):
        device = self.device
        self.device = None
        torch.save(self, open(path, 'wb'))
        self.device = device
        print('Saved!')

    @staticmethod
    def load(path):
        model = torch.load(path,
                    map_location=lambda storage, location: storage)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.device = torch.device(dev)
        #model.net.device_str = dev
        return model

    def train(self, train_loader, test_loader, epochs = 2):
        # Train classifier
        #use_cuda = False
        #device = torch.device("cuda" if use_cuda else "cpu")
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader, epoch)

    def _sample_mask(self):
        """
            Randomly sample a rectangular mask
        """
        W, H   = self.input_size
        w, h   = self.mask_size
        i = np.random.choice(self.I)
        j = np.random.choice(self.J)
        S = np.s_[i:min(i+h,H),j:min(j+w,W)]
        mask = torch.zeros(W,H).to(self.device) # FIXME: SHOULD BE TRUE ZERO
        mask[S[0],S[1]] = 1
        return mask, S

    def _get_knn_label_freqs(self, x, mask = None, X = None, Y = None, return_nn = False):
        """
            Given query (masked) vectors x, and mask S, returns class labels of knn
            with respect to distance induced by mask S, (i.e., ignoring rest of image)

            - X should be unnormalized, mask will, be applied directly
            - X_s should be masked and unnormalized
        """
        W, H = self.input_size
        index = faiss.IndexFlatL2(W*H)   # build the index
        #pdb.set_trace()
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add((X.squeeze() * np.tile(mask,(X.shape[0],1,1))).reshape(-1,W*H))                  # add vectors to the index
        D, Idxs = index.search(x.view(-1,W*H).cpu().numpy(), self.knn)     # actual search
        ## FIXME: this looks like a bug from numpy
        if Idxs.shape[0] < Idxs.shape[1]:
            # If # rows is too small, can't slice Y for some reason. Add dummy rows.
            diff = 30
            Classes = Y[np.vstack((Idxs,np.zeros((diff, self.knn), dtype=int)))][:Idxs.shape[0],:]
        else:
            Classes = Y[Idxs]
        #pdb.set_trace()
        Freqs = np.apply_along_axis(lambda a: np.histogram(a, bins=range(self.nclasses+1))[0], 1, Classes)#.numpy())
        assert Freqs.shape[0] == x.shape[0]
        if return_nn:
            return Freqs, Idxs
        else:
            return Freqs


    def _get_reference_data(self,loader):
        if self.task == 'mnist':
            #pdb.set_trace()
            if loader.dataset.train:
                X_full = (loader.dataset.train_data/255).numpy().astype('float32')
                Y_full = loader.dataset.train_labels.numpy()
            else:
                X_full = (loader.dataset.test_data/255).numpy().astype('float32')
                Y_full = loader.dataset.test_labels.numpy()
        elif self.X is not None and self.Y is not None:
            # Reference data has been precomputed outside this function
            X_full, Y_full = self.X, self.Y
        return X_full, Y_full

    def train_epoch(self, train_loader, epoch):
        self.net.train()

        X_full, Y_full = self._get_reference_data(train_loader)
        W, H   = self.input_size
        w, h   = self.mask_size

        ntrain = len(train_loader.sampler.indices) # len(test_loader.dataset is wrong if using subsetsampler!)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, _ = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            mask, S = self._sample_mask()

            # Mask Inputs
            if self.task == 'mnist':
                X_s = mnist_unnormalize(data)*mask.view(1,1,W,H).repeat(data.shape[0], 1, 1, 1)
            elif self.task in ['hasy','leafsnap']:
                X_s = data*mask.view(1,1,W,H).repeat(data.shape[0], 1, 1, 1)

            # Compute Nearest neighbors with this mask
            freqs = self._get_knn_label_freqs(X_s, mask = mask, X=X_full,Y=Y_full)
            target = (torch.FloatTensor(freqs)/self.knn).to(self.device)

            #pdb.set_trace()
            if self.task == 'mnist':
                output = self.net(mnist_normalize(X_s))
            elif self.task in ['hasy','leafsnap']:
                output = self.net(X_s)

            loss = F.binary_cross_entropy(output, target)

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),ntrain,
                    100. * batch_idx / len(train_loader), loss.item()))
            #break

    def test(self, test_loader, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0

        # FIXME: DO we really need test_loader here? FOr hasy, I'm using same
        X_full, Y_full = self._get_reference_data(test_loader)
        W, H   = self.input_size
        w, h   = self.mask_size

        if type(test_loader.sampler) is SequentialSampler:
            ntest = len(test_loader.dataset)
        elif type(test_loader.sampler) is SubsetRandomSampler:
            ntest = len(test_loader.sampler.indices)
        else:
            pdb.set_trace()
            raise ValueError("Wrong sampler")
        #ntest = len(test_loader.sampler.indices) # len(test_loader.dataset is wrong if using subsetsampler!)

        with torch.no_grad():
            for (data, target) in tqdm(test_loader):
                #print(idx)
                data, target = data.to(self.device), target.to(self.device)
                mask, S = self._sample_mask()

                # Mask Inputs
                if self.task == 'mnist':
                    X_s = mnist_unnormalize(data)*mask.view(1,1,W,H).repeat(data.shape[0], 1, 1, 1)
                elif self.task in ['hasy','leafsnap']:
                    X_s = data*mask.view(1,1,W,H).repeat(data.shape[0], 1, 1, 1)

                #pdb.set_trace()
                # Compute Nearest neighbors with this mask
                freqs = self._get_knn_label_freqs(X_s, mask = mask, X=X_full,Y=Y_full)
                target = (torch.FloatTensor(freqs)/self.knn).to(self.device)

                #pdb.set_trace()
                if self.task == 'mnist':
                    output = self.net(mnist_normalize(X_s))
                elif self.task in ['hasy','leafsnap']:
                    output = self.net(X_s)
                #loss = F.binary_cross_entropy(output, target)

                test_loss += F.binary_cross_entropy(output, target, size_average = False).item()
                #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                #correct = 0
                # Retrieve correct from options
                #correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= ntest
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct,ntest, 100. * correct / ntest))
        self.samples_write(data, S, X_s, mask, output, target, epoch, X_full, Y_full)

    def samples_write(self, x, S, xs, mask, pred, target, epoch, X_full = None, Y_full = None):
        W, H   = self.input_size
        w, h   = self.mask_size
        examples = 10
        fig = plt.figure(figsize=(15, 2*examples))
        gs = gridspec.GridSpec(1, 3, width_ratios = [1.5,1,2])
        gsp = gridspec.GridSpecFromSubplotSpec(examples, 2, subplot_spec=gs[0], wspace=0.1, hspace=0.2)
        gsm = gridspec.GridSpecFromSubplotSpec(examples, 1, subplot_spec=gs[1], wspace=0.1, hspace=0.2)
        gsb = gridspec.GridSpecFromSubplotSpec(examples, 2, subplot_spec=gs[2],wspace=0.1, hspace=0.2)
        ii = list(range(W)[S[0]])[0]
        jj = list(range(H)[S[1]])[0]
        titles = ['Input', 'Masked Attribute', 'True Class Freq', 'Pred Class Freq']

        for i in range(examples):
            # Plot original + mask boundary
            axp1 =  plt.subplot(gsp[i,0])
            axp1.imshow(x[i].reshape(W,H),  aspect="auto", cmap='Greys')
            axp1.axis('off')
            rect = patches.Rectangle((jj-0.01,ii-0.01),w,h,linewidth=1,edgecolor='r',facecolor='none')
            axp1.add_patch(rect)
            # Plot masked input
            axp2 =  plt.subplot(gsp[i,1])
            axp2.axis('off')
            axp2.imshow(xs[i].cpu().numpy().squeeze(),  aspect="auto", cmap='Greys')

            # PLot nearest neuighbors
            # Plot masked input
            # Compute Nearest neighbors with this mask
            freqs, NN = self._get_knn_label_freqs(xs, mask = mask, X=X_full,Y=Y_full, return_nn = True)
            gsgrid =  gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gsm[i], wspace=0.00, hspace=0.00)

            samples = X_full[NN[i][:16],:]
            #samples = self.forward(x)
            #samples = samples.data.cpu().numpy()[:16]
            #gs = gridspec.GridSpec(4, 4)
            #gs.update(wspace=0.05, hspace=0.05)
            for j, sample in enumerate(samples):
                ax = plt.subplot(gsgrid[j])
                plt.axis('off')
                # if i == 0 and j ==0:
                #     plt.suptitle('Nearest Neighbors')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(W,H), cmap='Greys')
            #axp2.axis('off')
            #axp2.imshow(xs[i].numpy().squeeze(),  aspect="auto")

            # Plot histograms
            axb1, axb2 =  plt.subplot(gsb[i,0]),  plt.subplot(gsb[i,1])
            axb1.bar(range(self.nclasses), target[i].detach().cpu().numpy().squeeze())
            #axb1.bar(range(10), freqs[i])
            axb2.bar(range(self.nclasses), pred[i].detach().cpu().numpy().squeeze())
            for ax in [axb1, axb2]:
                ax.set_ylim(0,1)
                if self.nclasses < 20:
                    ticks, labels = range(self.nclasses), range(self.nclasses)
                else:
                    ticks, labels = [], []
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
            axb2.set_yticklabels([])
            if i == 0:
                axp1.set_title('Input')
                axp2.set_title('Masked Attrib')
                axb1.set_title('True Class Freq')
                axb2.set_title('Pred Class Freq')

        if not os.path.exists('../out/mask_' + self.task):
            os.makedirs('../out/mask_{}/'.format(self.task))
        plt.savefig('../out/mask_{}/{}.png'.format(self.task, str(epoch).zfill(3)), bbox_inches='tight')
        plt.show()


class masked_text_classifier():
    def __init__(self, vocab, langs, task = 'ets', hidden_dim = 150,
                optim = 'adam', mask_type = 'disjoint',
                missing_method = 'zero', padding = 0, knn = 20, #image_size = (28,28),
                mask_size = 5, log_interval = 10, X = None, Y = None, **kwarg):
        super(masked_text_classifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = log_interval
        self.plot_interval = 1000
        #self.padding = padding
        self.knn = knn
        self.missing_method = 'zero'
        self.mask_size = mask_size
        self.mask_type = mask_type
        self.task = task
        #self.X, self.Y = X, Y
        self.emb_dim = vocab.vectors.shape[1]
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.langs = langs
        self.nclasses = len(langs)
        self.input_size = None # IT's variable
        # Need to pass vocab, etc to LSTM classifier
        #self.net = LSTMClassifier(self.emb_dim, hidden_dim, len(vocab), len(langs), vocab = vocab).to(self.device)
        self.net = NGramCNN(vocab = vocab, ngram = self.mask_size,
                            nclasses = self.nclasses, final_nonlin='sigmoid').to(self.device)

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)

        ## Initialize mask corner points
        # I, J contain valid left and top init pixels
        # if mask_type == 'disjoint':
        #     # Partition input into equallty sized, disjoint squares
        #     self.I = list(range(0, self.input_size[0], mask_size[0]))
        #     self.J = list(range(0, self.input_size[1], mask_size[1]))
        # elif mask_type == 'overlapping':
        #     # Overlapping squares
        #     self.I = list(range(self.padding, self.input_size[0] - self.mask_size[0]- self.padding))
        #     self.J = list(range(self.padding, self.input_size[1] - self.mask_size[1] - self.padding))
        # print('I,J:', self.I, self.J)

    def __call__(self, x):
        return self.net(x)

    def eval(self):
        self.net.eval()

    @staticmethod
    def load(path):
        if 'gpu' in path:
            model = torch.load(path,
                        map_location=lambda storage, location: storage)
        else:
            model = torch.load(path)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.device = torch.device(dev)
        #model.net.device_str = dev
        return model

    def save(self, path):
        # Don't want to save potentially huge dicts of freqs
        self.train_label_freqs = None
        self.test_label_freqs  = None
        device = self.device
        self.device = None
        torch.save(self, open(path, 'wb'))
        self.device = device
        print('Saved!')

    def train(self, train_loader, test_loader, epochs = 2):
        # Train classifier
        #use_cuda = False
        #device = torch.device("cuda" if use_cuda else "cpu")
        #pdb.set_trace()
        print('Computing ngram label histograms')
        self.train_label_freqs = get_ngram_label_freqs(train_loader.dataset,
                        n = self.mask_size,
                        text_field  = train_loader.dataset.fields['text'],
                        label_field = train_loader.dataset.fields['label'],
                        density = True)
        self.test_label_freqs = get_ngram_label_freqs(test_loader.dataset,
                        n = self.mask_size,
                        text_field  = train_loader.dataset.fields['text'],
                        label_field = train_loader.dataset.fields['label'],
                        density = True)
        #pdb.set_trace()

        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader, epoch)

    def _sample_mask(self, max_len, lens):
        """
            Randomly sample an ngram mask for a batch of size b x max_len,
            but for which the true lengths of seqs are given by lens
        """
        mask = torch.zeros(len(lens), max_len).byte().to(self.device)
        i_s  = [np.random.choice(np.maximum(1, n - self.mask_size)) for n in lens]
        for b,i in enumerate(i_s):
            mask[b,i:i+self.mask_size].fill_(True)
        S = [range(i,i+self.mask_size) for i in i_s]
        print(S)
        print("Added this checkpoint to check that fill_ works!")
        pdb.set_trace()
        return mask, S

    def _get_target_histogram(self, label_freqs, grams):
        hists = []
        for t in grams.cpu().numpy():
            key = tuple(t)
            hist = label_freqs[key] if key in label_freqs else np.ones(self.nclasses)/self.nclasses
            hists.append(hist)
        targets = torch.from_numpy(np.vstack(hists)).float().to(self.device)
        return targets

    # def _get_reference_data(self, train_loader):
    #     # if self.task == 'ets':
    #     #     X_full = (train_loader.dataset.train_data/255).numpy().astype('float32')
    #     #     Y_full = train_loader.dataset.train_labels.numpy()
    #     # if self.task == 'hasy':
    #     #     X_full, Y_full = self.X, self.Y
    #     return X_full, Y_full

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        #label_freqs = self.label_freqs

        ntrain = len(train_loader.dataset)

        for batch_idx, batch in enumerate(train_loader):
            #data, _ = data.to(self.device), target.to(self.device)
            (inputs, lengths), targets = batch.text, batch.label
            inputs, lengths, targets = inputs.to(self.device), lengths.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            mask, S = self._sample_mask(inputs.shape[1], lengths)
            grams = torch.masked_select(inputs, mask).view(inputs.shape[0], -1)
            target_histograms = self._get_target_histogram(self.train_label_freqs, grams)
            output = self.net(grams)
            if batch_idx == 0:
                print(grams[0])
                print(target_histograms[0])

            loss = F.binary_cross_entropy(output, target_histograms)

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch),ntrain,
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, test_loader, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0

        ntest = len(test_loader.dataset)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                (inputs, lengths), targets = batch.text, batch.label
                inputs, lengths, targets = inputs.to(self.device), lengths.to(self.device), targets.to(self.device)
                mask, S = self._sample_mask(inputs.shape[1], lengths)
                grams = torch.masked_select(inputs, mask).view(inputs.shape[0], -1)
                target_histograms = self._get_target_histogram(self.test_label_freqs, grams)
                output = self.net(grams)
                print(output[0])
                print(target_histograms[0])
                test_loss += F.binary_cross_entropy(output, target_histograms)

        test_loss /= ntest
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, ntest, 100. * correct / ntest))
        #self.samples_write(data, S, X_s, mask, output, target, epoch, X_full, Y_full)

    def samples_write(self, x, S, xs, mask, pred, target, epoch, X_full = None, Y_full = None):
        W, H   = self.input_size
        w, h   = self.mask_size
        examples = 10
        fig = plt.figure(figsize=(15, 2*examples))
        gs = gridspec.GridSpec(1, 3, width_ratios = [1.5,1,2])
        gsp = gridspec.GridSpecFromSubplotSpec(examples, 2, subplot_spec=gs[0], wspace=0.1, hspace=0.2)
        gsm = gridspec.GridSpecFromSubplotSpec(examples, 1, subplot_spec=gs[1], wspace=0.1, hspace=0.2)
        gsb = gridspec.GridSpecFromSubplotSpec(examples, 2, subplot_spec=gs[2],wspace=0.1, hspace=0.2)
        ii = list(range(W)[S[0]])[0]
        jj = list(range(H)[S[1]])[0]
        titles = ['Input', 'Masked Attribute', 'True Class Freq', 'Pred Class Freq']

        for i in range(examples):
            # Plot original + mask boundary
            axp1 =  plt.subplot(gsp[i,0])
            axp1.imshow(x[i].reshape(W,H),  aspect="auto", cmap='Greys')
            axp1.axis('off')
            rect = patches.Rectangle((jj-0.01,ii-0.01),w,h,linewidth=1,edgecolor='r',facecolor='none')
            axp1.add_patch(rect)
            # Plot masked input
            axp2 =  plt.subplot(gsp[i,1])
            axp2.axis('off')
            axp2.imshow(xs[i].cpu().numpy().squeeze(),  aspect="auto", cmap='Greys')

            # PLot nearest neuighbors
            # Plot masked input
            # Compute Nearest neighbors with this mask
            freqs, NN = self._get_knn_label_freqs(xs, mask = mask, X=X_full,Y=Y_full, return_nn = True)
            gsgrid =  gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gsm[i], wspace=0.00, hspace=0.00)

            samples = X_full[NN[i][:16],:]
            #samples = self.forward(x)
            #samples = samples.data.cpu().numpy()[:16]
            #gs = gridspec.GridSpec(4, 4)
            #gs.update(wspace=0.05, hspace=0.05)
            for j, sample in enumerate(samples):
                ax = plt.subplot(gsgrid[j])
                plt.axis('off')
                # if i == 0 and j ==0:
                #     plt.suptitle('Nearest Neighbors')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(W,H), cmap='Greys')
            #axp2.axis('off')
            #axp2.imshow(xs[i].numpy().squeeze(),  aspect="auto")

            # Plot histograms
            axb1, axb2 =  plt.subplot(gsb[i,0]),  plt.subplot(gsb[i,1])
            axb1.bar(range(self.nclasses), target[i].detach().cpu().numpy().squeeze())
            #axb1.bar(range(10), freqs[i])
            axb2.bar(range(self.nclasses), pred[i].detach().cpu().numpy().squeeze())
            for ax in [axb1, axb2]:
                ax.set_ylim(0,1)
                if self.nclasses < 20:
                    ticks, labels = range(self.nclasses), range(self.nclasses)
                else:
                    ticks, labels = [], []
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
            axb2.set_yticklabels([])
            if i == 0:
                axp1.set_title('Input')
                axp2.set_title('Masked Attrib')
                axb1.set_title('True Class Freq')
                axb2.set_title('Pred Class Freq')

        if not os.path.exists('out/mask_' + self.task):
            os.makedirs('out/mask_{}/'.format(self.task))
        plt.savefig('out/mask_{}/{}.png'.format(self.task, str(epoch).zfill(3)), bbox_inches='tight')
        plt.show()


from collections import defaultdict
from nltk import ngrams

def get_ngram_label_freqs(dataset, n = 3, text_field = None, label_field=None,
    device = -1, density = False):
    """
        If fields provided, will use them to map to indices
    """
    classes     = set([])
    class_freqs = defaultdict(list)

    i = 0
    print('Getting ngram counts....')
    for example in tqdm(dataset):
        grams = ngrams(example.text, n)
        for gram in grams:
            if text_field:
                # by using numericalize instead of process we avoid padding
                key = text_field.numericalize(([gram],[]), device = device)[0][0]
                key = tuple(key.numpy())
            else:
                key = gram
            if label_field:
                value =label_field.process([example.label], device = device, train = False).item()
            else:
                value = example.label
            class_freqs[key].append(value)
            classes.update([value])

        i +=1
        # if i==1000:
        #     break

    classes = range(len(label_field.vocab.itos)) if label_field else sorted(list(classes))

    print('Computing histograms...')
    # Do this with tensor.histc - first stack, compute, slice?
    for k,v in tqdm(class_freqs.items()):
        class_freqs[k] = np.histogram(v, bins=range(len(classes)+1), density = density)[0] #, 1, Classes)#.numpy())

    return class_freqs
