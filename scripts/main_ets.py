import os
import pathlib
import pdb
import pickle
import argparse


import matplotlib
matplotlib.use('Agg')

import torch
from torchtext import data as tntdata
import spacy
spacy_en = spacy.load('en')

from utils import generate_dir_names
from models import ets_classifier, masked_text_classifier

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_ets_data(data_root = None, batch_sizes = (32, 256, 256),
    embeddings = 'glove-100', debug = False):

    # Data paths etc
    if data_root is None:
        root = os.path.dirname(os.getcwd())
        data_root = pathlib.Path(root + '/data/ETS')

    data_dir = os.path.join(data_root, 'data/text/')

    if embeddings == 'glove-100':
        embname = "glove.6B.100d"
    elif embeddings == 'glove-300':
        embname = "glove.6B.300d"
    else:
        raise ValueError("Unrecognized embeddings")

    # Define filds for torchtext
    TEXT = tntdata.Field(
        sequential=True,
        tokenize = tokenizer, #tntdata.get_tokenizer('spacy'),
        lower=True,
        batch_first = True,
        include_lengths=True,
        #fix_length = 200, # Shorter string will be padded
        init_token='<SOS>', eos_token='<EOS>'
    )

    #LABEL = tntdata.Field(sequential=False, use_vocab=False)
    LABEL = tntdata.LabelField()

    # Get data
    train_ext = 'dev.tsv' if debug else 'train.tsv'
    train, val, test = tntdata.TabularDataset.splits(
        path = os.path.join(data_root,'processed/'),
        train = train_ext, validation = 'dev.tsv', test = 'test.tsv',
        format='tsv', skip_header = True,
        fields=[(None, None), ('text', TEXT),
                ('label', LABEL), (None, None)])   # We ignore fist and last fields (id, score)

    # Get dynamic batch loaders for each fold (these take care of padding, etc)
    train_loader, val_loader, test_loader = tntdata.BucketIterator.splits(
        (train, val, test), sort_key=lambda x: len(x.text),
        batch_sizes=batch_sizes, device=-1, sort_within_batch=True, repeat = False)

    # Build vocabs
    TEXT.build_vocab(train, vectors = embname)
    LABEL.build_vocab(train)
    vocab = TEXT.vocab
    langs = LABEL.vocab.itos

    print('Dataset size: {}/{}/{} (Train, valid, test)'.format(
        len(train), len(val), len(test)))
    print('Vocabulary size: ', len(vocab))
    print('Number of languages: {}, {}'.format(len(langs),' / '.join(langs)))
    print('Embedding size: ', vocab.vectors.shape[1])

    return train_loader, val_loader, test_loader, train, val, test, vocab, langs

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

    # Model parameters - Classifier
    parser.add_argument('--num_layers_clf', type=int, default=2, help='number of hidden layers for classifier [default: 2]')
    parser.add_argument('--dropout_clf', type=float, default=.5, help='droput for classifier model')
    parser.add_argument('--weight_decay_clf', type=float, default=1e-06, help='droput for classifier model')
    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden dim [default: 300]')
    parser.add_argument('--embdim', type=int, default=300, help='word embedding layer dim [default: 300]')
    parser.add_argument('--ngram', type=int, default=5, help='ngram size [default: 5]')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 64]')
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

def get_relabeled_loader(model, dataset, classes, batch_size = 128, device = -1):
    """
        Relabeled dataset with predictions of model, retrieve as loader
    """
    # 1. Loader to label - couldnt find any other way to keep track of indices
    # but to use batch_size = 1. FIXME: Find a batch-version to do it.
    loader_unshuffled = tntdata.Iterator(dataset = dataset,
    shuffle = False, sort = False, repeat = False, batch_size=1, device=device) # Train = false causes shuffle, repeat = false and sort = True

    preds = model.label_datatset(loader_unshuffled)

    #train_loader.dataset.train_labels = train_pred
    for (i,pr) in enumerate(preds):
        #print(i, pr, train_loader.dataset.examples[i].label, langs[pr])
        dataset.examples[i].label = classes[pr]

    loader_relabeled = tntdata.BucketIterator(
                    dataset = dataset, batch_size=batch_size,
                    sort_key=lambda x: len(x.text),
                    sort_within_batch=True, repeat = False,device=device) # Train = false causes shuffle, repeat = false and sort = True
    #pdb.set_trace()
    # Get dynamic batch loaders for each fold (these take care of padding, etc)
    # train_loader, val_loader, test_loader = tntdata.BucketIterator.splits(
    #     (train, val, test), sort_key=lambda x: len(x.text),
    #     batch_sizes=batch_sizes, device=-1, sort_within_batch=True, repeat = False)
    #
    return loader_relabeled




def main():
    args = parse_args()
    model_path, log_path, results_path = generate_dir_names('ets', args)

    # Load Data
    print('Loading data...', end='')
    train_loader, val_loader, test_loader, train_tds, val_tds, test_tds, vocab, langs = get_ets_data(
        data_root = os.path.realpath('../../data/ETS'),
        embeddings = 'glove-{}'.format(str(args.embdim)),
        batch_sizes = (args.batch_size, 256, 256),
        debug = args.debug)
    print('done!')

    # Train or load classifier
    if args.train or (not os.path.isfile(os.path.join(model_path, "classif.pth"))):
        print('Training classifier from scratch')
        model = ets_classifier(vocab, langs,
                               weight_decay = args.weight_decay_clf,
                               hidden_dim = args.hidden_dim,
                               num_layers = args.num_layers_clf,
                               dropout = args.dropout_clf,
                               lr = args.lr,
                               use_cuda = args.cuda)
        model.train(train_loader, val_loader, epochs = args.epochs_classif)
        model.save(os.path.join(model_path, "classif.pth"))
    else:
        print('Loading pre-trained classifier')
        model = torch.load(os.path.join(model_path, "classif.pth"))
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: test data doesn't have labels! It's useless
    # TODO: Split train into actual val and train
    model.test(test_loader)

    ### Label training examples with ets_classifier
    #train_loader_relabeled = get_relabeled_loader(model, train_tds, langs, args.batch_size)
    #test_loader_relabeled  = get_relabeled_loader(model, val_tds, langs, args.batch_size)
    # Debug
    train_loader_relabeled = train_loader
    test_loader_relabeled  = test_loader

    ### Train meta model
    mask_type = 'overlapping'

    mask_model = masked_text_classifier(
        vocab, langs, hidden_dim = args.hidden_dim, task = 'ets', optim = args.optim,
        mask_size = args.ngram, mask_type = mask_type, log_interval = 10)

    mask_model.train(train_loader_relabeled, test_loader_relabeled, epochs = args.epochs_meta)

    mask_model.save(os.path.join(model_path, "mask_model_{}.pth".format(args.ngram)))


if __name__ == '__main__':
    main()
