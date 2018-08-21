import os
import pathlib
import pdb
import pickle
import argparse


import matplotlib as mpl
if mpl.get_backend() == 'Qt5Agg':
    # Means this is being run in server, need to modify backend
    mpl.use('Agg')

import torch


from src.utils import generate_dir_names
from src.datasets import load_ets_data
from src.models import text_classifier, masked_text_classifier
from src.explainers import MCExplainer


def parse_args():
    parser = argparse.ArgumentParser(add_help=False,
                                     description='Interpteratbility robustness evaluation on ETS dataset')

    parser.add_argument('--train_classif', action='store_true', default=False,
                        help='Whether or not to (re)train classifier model')
    parser.add_argument('--train_meta', action='store_true', default=False,
                        help='Whether or not to (re)train meta masking model')

    # Model parameters - Classifier
    parser.add_argument('--num_layers_clf', type=int, default=2,
                        help='number of hidden layers for classifier [default: 2]')
    parser.add_argument('--dropout_clf', type=float,
                        default=.5, help='droput for classifier model')
    parser.add_argument('--weight_decay_clf', type=float,
                        default=1e-06, help='droput for classifier model')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='hidden dim [default: 300]')
    parser.add_argument('--embdim', type=int, default=300,
                        help='word embedding layer dim [default: 300]')
    parser.add_argument('--ngram', type=int, default=5,
                        help='ngram size [default: 5]')

    # Model parameters - Meta-learner
    parser.add_argument('--attrib_type', type=str, choices=['overlapping'],
                        default='overlapping', help='type of attribute')
    parser.add_argument('--attrib_width', type=int, default=7,
                        help='width of attribute region [default: 7]')
    parser.add_argument('--attrib_height', type=int, default=7,
                        help='height of attribute region [default: 7]')
    parser.add_argument('--attrib_padding', type=int, default=4,
                        help='Padding around input image to define attributes')

    # Explainer
    parser.add_argument('--mce_reg_type', type=str, choices=['exp', 'quadratic'],
                        default='exp', help='Type of regularization for explainer scoring function objective')
    parser.add_argument('--mce_alpha', type=float,
                        default=1.0, help='Alpha parameter for explainer scoring function')
    parser.add_argument('--mce_p', type=int, default=2,
                        help='P power for explainer scoring function')
    parser.add_argument('--mce_plottype', type=str, choices=['treemap', 'bar'],
                        default='treemap', help='Plot type of class entailment diagram')
    # device
    parser.add_argument('--cuda', action='store_true',
                        default=False, help='enable the gpu')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')

    # learning
    parser.add_argument('--optim', type=str, default='adam',
                        help='optim method [default: adam]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs_classif', type=int, default=10,
                        help='number of epochs for train [default: 10]')
    parser.add_argument('--epochs_meta', type=int, default=10,
                        help='number of epochs for train [default: 10]')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training [default: 64]')
    parser.add_argument('--objective', default='cross_entropy',
                        help='choose which loss objective to use')

    # paths
    parser.add_argument('--model_path', type=str,
                        default='models', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='src/out',
                        help='where to dump model config and epoch stats')
    parser.add_argument('--log_path', type=str, default='src/log',
                        help='where to dump training logs  epoch stats (and config??)')

    # data loading
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers for data loader')


    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def get_relabeled_loader(model, dataset, classes, batch_size=128, device=-1):
    """
        Relabeled dataset with predictions of model, retrieve as loader
    """
    # 1. Loader to label - couldnt find any other way to keep track of indices
    # but to use batch_size = 1. FIXME: Find a batch-version to do it.
    loader_unshuffled = tntdata.Iterator(dataset=dataset,
                                         shuffle=False, sort=False, repeat=False, batch_size=1, device=device)  # Train = false causes shuffle, repeat = false and sort = True

    preds = model.label_datatset(loader_unshuffled)

    #train_loader.dataset.train_labels = train_pred
    for (i, pr) in enumerate(preds):
        #print(i, pr, train_loader.dataset.examples[i].label, langs[pr])
        dataset.examples[i].label = classes[pr]

    loader_relabeled = tntdata.BucketIterator(
        dataset=dataset, batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True, repeat=False, device=device)  # Train = false causes shuffle, repeat = false and sort = True
    # pdb.set_trace()
    # Get dynamic batch loaders for each fold (these take care of padding, etc)
    # train_loader, val_loader, test_loader = tntdata.BucketIterator.splits(
    #     (train, val, test), sort_key=lambda x: len(x.text),
    #     batch_sizes=batch_sizes, device=-1, sort_within_batch=True, repeat = False)
    #
    return loader_relabeled


def main():
    args = parse_args()
    model_path, log_path, results_path = generate_dir_names('ets', args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print('Loading data...', end='')
    train_loader, val_loader, test_loader, train_tds, val_tds, test_tds, vocab, langs \
        = load_ets_data(
            data_root=os.path.realpath('data/processed/ets'),
            embeddings='glove-{}'.format(str(args.embdim)),
            batch_sizes=(args.batch_size, 256, 256),
            debug=args.debug)
    print('done!')
    class_names = langs

    ### TRAIN OR LOAD CLASSIFICATION MODEL TO BE EXPLAINED
    classif_path = os.path.join(model_path, "classif.pth")
    print(classif_path)
    if args.train_classif or (not os.path.isfile(classif_path)):
        print('Training classifier from scratch')
        clf = text_classifier(vocab, langs,
                               weight_decay=args.weight_decay_clf,
                               hidden_dim=args.hidden_dim,
                               num_layers=args.num_layers_clf,
                               dropout=args.dropout_clf,
                               lr=args.lr,
                               use_cuda=args.cuda)
        clf.train(train_loader, val_loader, epochs=args.epochs_classif)
        clf.save(os.path.join(model_path, "classif.pth"))
    else:
        print('Loading pre-trained classifier')
        #clf = #torch.load(os.path.join(model_path, "classif.pth"))
        clf = text_classifier.load(classif_path)
        #clf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #pdb.set_trace()
    # NOTE: test data doesn't have labels! It's useless
    # TODO: Split train into actual val and train
    clf.test(test_loader)

    ### TRAIN OR LOAD META-MODEL OF MASKED INPUTS
    metam_path = os.path.join(model_path, "mask_model_{}.pth".format(args.ngram))
    if args.train_meta or (not os.path.isfile(metam_path)):
        print('Training meta masking model from scratch')
        # Label training examples with ets_classifier
        #train_loader_relabeled = get_relabeled_loader(model, train_tds, langs, args.batch_size)
        #test_loader_relabeled  = get_relabeled_loader(model, val_tds, langs, args.batch_size)
        # Debug
        train_loader_relabeled = train_loader
        test_loader_relabeled = test_loader

        # Train meta model
        mask_type = 'overlapping'

        mask_model = masked_text_classifier(
            vocab, langs, hidden_dim=args.hidden_dim, task='ets', optim=args.optim,
            mask_size=args.ngram, mask_type=mask_type, log_interval=10)

        mask_model.train(train_loader_relabeled,
                         test_loader_relabeled, epochs=args.epochs_meta)

        mask_model.save(os.path.join(
            model_path, "mask_model_{}.pth".format(args.ngram)))
    else:
        print('Loading pre-trained meta masking model')
        mask_model = masked_text_classifier.load(metam_path)
        mask_model.net = mask_model.net.to(device)
        pdb.set_trace()
        #mask_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### EXPLAIN SOME INSTANCES

    Explainer = MCExplainer(clf, mask_model, classes = class_names,
                      reg_type = args.mce_reg_type,
                      crit_alpha = args.mce_alpha,
                      crit_p = args.mce_p,
                      plot_type = args.mce_plottype)

    # Grab a batch for experiments
    batch = next(iter(test_loader))
    batch_x, batch_x_lens = batch.text
    batch_y = batch.label
    batch_x, batch_x_lens= batch_x.to(device), batch_x_lens.to(device)
    batch_y = batch_y.to(device)

    idx = 15
    x_len = batch_x_lens[idx:idx+1]
    x  = batch_x[idx:idx+1]
    fx = clf(x, x_len)
    p, pred = fx.max(1)
    p = torch.softmax(fx).max().item()
    #print(classes[batch_y[idx].item()])
    e = Explainer.explain(x, pred.item(), verbose = 0 , show_plot = 1)


if __name__ == '__main__':
    main()
