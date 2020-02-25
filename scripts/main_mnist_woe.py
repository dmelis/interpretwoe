import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
if matplotlib.get_backend() == 'Qt5Agg':
    # Means this is being run in server, need to modify backend
    matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import torch
import numpy as np

# Local Imports
from src.models import image_classifier
from src.models import masked_image_classifier
from src.utils import generate_dir_names
from src.explainers import MCExplainer
from src.datasets import load_mnist_data, load_full_dataset
from src.models import mnist_unnormalize

from src.woe_utils import restore_model_from_file

def parse_args():
    ### Local ones
    parser = argparse.ArgumentParser(add_help=False,
        description='Interpteratbility robustness evaluation on MNIST')

    parser.add_argument('--train_classif', action='store_true', default=False,
                        help='Whether or not to (re)train classifier model')
    parser.add_argument('--train_meta', action='store_true', default=False,
                        help='Whether or not to (re)train meta masking model')

    # Meta-learner
    parser.add_argument('--attrib_type', type=str, choices = ['overlapping'],
                        default='overlapping', help='type of attribute')
    parser.add_argument('--attrib_width', type=int, default=7,
                        help='width of attribute region [default: 7]')
    parser.add_argument('--attrib_height', type=int, default=7,
                        help='height of attribute region [default: 7]')
    parser.add_argument('--attrib_padding', type=int, default=4,
                        help='Padding around input image to define attributes')

    # Explainer
    parser.add_argument('--paradigm', type=str, choices = ['likelihood', 'hybrid', 'classification'],
                            default='likelihood', help='Probabilistic explanation paradigm')
    parser.add_argument('--mce_reg_type', type=str, choices=['decay','poly-centered'],
                        default='decay', help='Type of regularization for explainer scoring function objective')
    parser.add_argument('--mce_alpha', type=float,
                            default=1.0, help='Alpha parameter for explainer scoring function')
    parser.add_argument('--mce_p', type=int, default=2, help='P power for explainer scoring function')
    parser.add_argument('--mce_plottype', type=str, choices = ['treemap', 'bar'],
                            default='treemap', help='Plot type of class entailment diagram')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode' )
    parser.add_argument('--seed', type = int, default=2018, help='set seed. Choose -1 for no seed.' )

    # learning
    parser.add_argument('--optim', type=str, default='adam', help='optim method [default: adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs_classif', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--epochs_meta', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')

    #paths
    parser.add_argument('--model_path', type=str, default='models/', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='src/out', help='where to dump model config and epoch stats')
    parser.add_argument('--log_path', type=str, default='src/log', help='where to dump training logs  epoch stats (and config??)')

    # data loading
    parser.add_argument('--num_workers' , type=int, default=4, help='num workers for data loader')


    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args



def main():
    args = parse_args()
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
    args.nclasses = 10
    classes = [str(i) for i in range(10)]
    model_path, log_path, results_path = generate_dir_names('mnist', args)
    home_dir = os.path.expandvars('$HOME')

    ### LOAD DATA
    train_loader, valid_loader, test_loader, train_tds, test_tds = load_mnist_data(
                        batch_size=args.batch_size,num_workers=args.num_workers
                        )

    ### TRAIN OR LOAD CLASSIFICATION MODEL TO BE EXPLAINED
    classif_path = os.path.join(model_path, "classif.pth")
    if args.train_classif or (not os.path.isfile(classif_path)):
        print('Training classifier from scratch')
        clf = image_classifier(task='mnist',optim = args.optim, use_cuda = args.cuda)
        clf.train(train_loader, valid_loader, epochs = args.epochs_classif)
        clf.save(classif_path)
    else:
        print('Loading pre-trained classifier')
        clf = image_classifier.load(classif_path)
        clf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.test(test_loader)

    ### TRAIN OR LOAD WOE ESTIMATOR
    mask_size = (args.attrib_width, args.attrib_height)
    #woe_model_path = os.path.join(model_path, "woe_model_{}x{}.pth".format(*mask_size))
    woe_model_path  = os.path.join(home_dir, 'workspace/normalizing_flows/results/maf/blocks_rnd/B5_H2_128_lr1e-5/best_model_checkpoint.pt')
    if args.train_meta or (not os.path.isfile(woe_model_path)):
        raise NotImplemented("need to add this - or put in a different script an expect trained LL/meta model?")
        print('Training meta masking model from scratch')
        # Label training examples with mnist_classifier
        # TODO: Before I was reloading dataset with shuffle=False. But it seems this
        # is not necessary, since I assign it to
        train_loader_unshuffled = torch.utils.data.dataloader.DataLoader(
                            train_tds, shuffle=False, batch_size=args.batch_size)
        train_pred = clf.label_datatset(train_loader_unshuffled)
        train_loader.dataset.train_labels = train_pred

        X_full, Y_full = load_full_dataset(train_loader_unshuffled, to_numpy=True,
                            max_examples =None, unnormalize = mnist_unnormalize)
        # Train meta model
        woe_model = masked_image_classifier(task = 'mnist', X = X_full, Y = Y_full,
                                             optim = args.optim,
                                             mask_type = args.attrib_type,
                                             padding = args.attrib_padding,
                                             mask_size = mask_size)
        woe_model.train(train_loader, test_loader, epochs = args.epochs_meta)
        woe_model.save(metam_path)
    else:
        print('Loading pre-trained meta masking model')
        woe_model, woe_args = restore_model_from_file(woe_model_path)
        #woe_model = masked_image_classifier.load(metam_path)
        #woe_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### EXPLAIN SOME INSTANCES
    Explainer = MCExplainer('mnist', clf, woe_model, classes = classes,
                      reg_type = args.mce_reg_type,
                      crit_alpha = args.mce_alpha,
                      crit_p = args.mce_p,
                      plot_type = args.mce_plottype)

    # Grab a batch for experiments
    batch_x, batch_y = next(iter(train_loader))
    idx = 3
    x  = batch_x[idx:idx+1]
    print(classes[batch_y[idx].item()])

    e = Explainer.explain(x, verbose = 0 , show_plot = 1)
    save_path = os.path.join(args.results_path, 'mnist/expl_id-{}_{}_{}_alpha-{}_p-{}.pdf'.format(
                    idx, args.paradigm, args.mce_reg_type, args.mce_alpha, args.mce_p))
    e.plot(save_path = save_path)


if __name__ == '__main__':
    main()
