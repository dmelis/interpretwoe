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


def make_latex_grid(y, ncol=8, padding=2,
              normalize=False, scale_each=False, pad_value=0):
    """Make a grid of images
    """
    cells = len(y)
    nc = min(ncol, cells)
    nr = int(math.ceil(float(cells) / nc))
    height, width = .5, .5
    #fig, axes = plt.subplots(nr, nc, figsize = (height*nr, width*nc))
    plt.figure(figsize = (height*nr, width*nc))
    gs1 = gridspec.GridSpec(nr, nc)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.

    for i in range(cells):
        #c = math.floor(i/nc)
        #r = i % nc
        ax = plt.subplot(gs1[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        if y[i].startswith('\Bowtie'):
        #except:
            continue
        ax.text(0.3, 0.4, r"$%s$" % y[i], size = 14)
    plt.show()


def generate_dir_names(dataset, args, make = True):
    # suffix = ''.format(
    #             args.theta_reg_type,
    #             args.h_type,
    #             args.nconcepts,
    #             args.theta_reg_lambda,
    #             args.h_sparsity,
    #             args.lr,
    #             )
    suffix = ''
    model_path     = os.path.join(args.model_path, dataset, suffix)
    log_path       = os.path.join(args.log_path, dataset, suffix)
    results_path   = os.path.join(args.results_path, dataset, suffix)

    if make:
        for p in [model_path, log_path, results_path]:
            if not os.path.exists(p):
                os.makedirs(p)

    return model_path, log_path, results_path


def animate_argopt(history, save_path = None):
    """
        Produce animation of argument optimization, e.g. when finding Factual Predicates.
        history should be a list of (iter, input, perturbation at time i, histogram at time i)
    """
    fig, ax = plt.subplots(1,3, figsize=(12,4))

    im1 = ax[0].imshow(history[0][1], animated=True, cmap=plt.get_cmap('YlOrRd'), interpolation = 'none')
    im2 = ax[1].imshow(history[0][2], animated=True, cmap=plt.get_cmap('YlOrRd'), interpolation = 'none')
    bar = ax[2].bar(range(10),history[0][3])

    ax[2].set_xticks(range(10))
    ax[2].set_xticklabels(range(10))
    ax[2].set_ylim(0,1)

    plt.subplots_adjust(top=0.75)
    ax[0].set_title('Input', fontsize = 20)
    ax[1].set_title('Perturbation', fontsize = 20)
    ax[2].set_title('Model Prediction', fontsize = 20)

    ax[0].grid(False)
    ax[1].grid(False)
    plt.tight_layout(pad=3.0)
    ax[0].axis('off')
    ax[1].axis('off')

    def init_imshow():
        """ CAn potential add here things that will show in all plots"""
        return [im1, im2]+[rect for rect in bar]

    def update_imshow(i, history):
        fig.suptitle('Iteration: {}'.format(history[i][0]), fontsize = 20)
        im1.set_array(history[i][1])
        im2.set_array(history[i][2])
        for rect, yi in zip(bar, range(len(range(10)))):
            rect.set_height(history[i][3][yi])
        return [im1, im2]+[rect for rect in bar]

    ani = animation.FuncAnimation(fig, update_imshow,init_func = init_imshow, fargs = (history,), frames = len(history), interval=50, blit=True)
    plt.close(fig) # Closes last one

    if save_path:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
    return ani

####################################################################
################ CONTRACTION / FIXED POINT TOOLS ###################
####################################################################

def eval_contractiveness(f, data_loader):
    total = 0
    f.eval()
    for batch_idx, (X,_) in enumerate(data_loader):
        #data, target = data.to(self.device), data.to(self.device)
        X_r = f(X)
        b = X.shape[0]
        X1, X2 = torch.split(X, [b/2, b/2], dim = 0)
        X1_r, X2_r = torch.split(X_r, [b/2, b/2], dim = 0)
        ratios = (X1_r - X2_r).view(int(b/2),-1).norm(dim=1) / \
                 (X1 - X2).view(int(b/2),-1).norm(dim=1)
        #print(ratios.sum())
        total += ratios.sum()

    return total/len(data_loader.dataset)



def find_fixed_point(f, x0, maxiters = 20, tol = 1e-04, plot = True):
    f.eval()
    xp = x0.clone()

    # if plot:
    #     fig, ax = plt.subplots(1, 2, figsize = (8,4))
    #     ax[0].imshow(x0.squeeze())
    #     im = ax[1].imshow(xp.squeeze())
    #     ax[0].axis('off')
    #     ax[1].axis('off')
    #     plt.show()

    for i in range(maxiters):
        xp_new = f(xp).reshape(1,28,28)
        delta = (xp_new - xp).norm().detach().numpy()
        if plot:
            clear_output(wait=True)
            #im.set_array(xp_new.detach().squeeze())
            fig, ax = plt.subplots(1, 2, figsize = (8,4))
            ax[0].imshow(x0.squeeze())
            im = ax[1].imshow(xp_new.detach().squeeze())
            ax[0].axis('off')
            ax[1].axis('off')
            ax[1].set_title('Iter {}, ||f(x(t)) - x(t)|| = {:.2e}'.format(i, delta))
            plt.show()
            #ax[1].draw()
            #plt.draw()
            #plt.imshow()
            #plt.axis('off')
            #plt.show()
        else:
            print(delta)
        xp = xp_new
        if delta < tol:
            break
    return xp


def contraction_plot(f, X, Y, maxit=20):
    """
            - examples: list of tuples [(k, x)]
    """
    #ncol = len(examples)
    ncol = X.shape[0]
    fig, ax = plt.subplots(1,ncol,figsize = (2*ncol, 2))
    for i in range(X.shape[0]):
        ax[i].imshow(X[i].squeeze())
        ax[i].axis('off')
    plt.show()

    for it in range(1, maxit+1):
        fig, ax = plt.subplots(1, ncol, figsize = (2*ncol,2))
        X_new = f(X)#.reshape(4,1,28,28)
        change = (X_new - X).reshape(ncol,-1).norm(dim=1)
        for i in range(ncol):
            ax[i].imshow(X_new[i].detach().squeeze())
            ax[i].set_title('{:4.2f}'.format(change[i].detach().numpy()))
            ax[i].axis('off')
        plt.show()

        printstr = ["Iter: {:d}".format(it)]
        for i in range(ncol):
            for j in range(i+1, ncol):
                lip  = ((X_new[j] - X_new[i]).norm()/(X[j] - X[i]).norm()).detach().numpy()
                printstr.append('Lip({},{}): {:4.2f}'.format(Y[i], Y[j], lip))

        print('\t'.join(printstr))
        #print("Iter: {0:d}\t Lip({1},{1}): {3:4.2f}\t Lip({2},{2}): {4:4.2f}"
        #"\t Lip({1},{2}): {5:4.2f}".format(it, classes[i], classes[j], lips[], lips[], lips))
        X = X_new

            #
        # lip_c1  = ((X_new[1] - X_new[0]).norm()/(X[1] - X[0]).norm()).detach().numpy()
        # lip_c2  = ((X_new[3] - X_new[2]).norm()/(X[3] - X[2]).norm()).detach().numpy()
        # lip_c12 = ((X_new[2] - X_new[0]).norm()/(X[2] - X[0]).norm()).detach().numpy()



    return X


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#
def attrib_barplot(hist, C, classes, topk=10, class_names=None, ax = None, sort_bars = True, pc =None):
    if ax is None:
        ax = plt.gca()

    # if sort_bars:
    #     #pdb.set_trace()
    #     arridx = hist.argsort()[::-1]
    #     hist = hist[arridx]
    #     #print(arridx, hist, classes)
    #     classes = classes[arridx]
    if topk:
        classes = classes[:topk]
        hist    = hist[:topk]
        nbars   = topk
    else:
        nbars   = nclass

    if class_names is not None:
        ticklabs =  [class_names[a] for a in classes]
    else:
        ticklabs  = [str(a) for a in classes]
    bar = ax.bar(ticklabs,hist)
    ax.set_xticks(range(nbars))
    ax.set_xticklabels(ticklabs)
    if sort_bars and not (topk < len(hist)) and (len(C) <= topk):
        # Draw separation lineself.
        #This only makes sense if (i) histogram is shown sorted, |C| < topk
        ax.axvline(x=len(C)-0.5, c ='red')
        if pc is not None:
            ax.text(len(C), 0.75*ax.get_ylim()[1], "P(C)/P(V)= %.2f" % pc, fontdict = {'color': 'red'},
            rotation=90, verticalalignment='center')#, transform = ax[ncol-1].transAxes)


def plot_text_attrib_entailment(x, xs, S, C, V, hist, plot_type = 'bar',
              title = None, show_cropped = True, vocab = None,
              sort_bars = True, topk = 10, class_names = None, cmap = 'Greys',
              ax = None, save_path = None):
    """
        Plot of text input, masked attribute and entailed probs
    """

    if ax is None:
        #ncol = 3 if show_cropped else 2
        ncol = 2
        fig, ax = plt.subplots(1, ncol, figsize = (10,4),
                            gridspec_kw = {'width_ratios':[2, 1]})
    else:
        ncol = len(ax)

    if type(V) is list:
        V = np.array(V)

    # Infer some aspects of input and mask
    nclass = len(hist)
    _, N   = x.shape
    classes = V #np.arange(nclass)
    ii = S[0]
    n  = len(S)

    window = 20
    #contex_idx = range(max(0,S[0] - window), n -1 + min(N, S[1] + window))
    contex_idx = range(max(0,S[0] - window), 1+ min(N-1,S[-1]+window))
    #print(S)
    #print(contex_idx)
    #print(len(contex_idx))
    #print(N, S, S[-1], S[-1]+window, contex_idx, len(contex_idx))
    input_crop = [vocab.itos[w.item()] for w in x.squeeze()[contex_idx]]
    if S[0] > window:
        input_crop[0] = '...'
    if S[1] < N - window:
        input_crop[-1] = '...'
    ngram = [vocab.itos[w.item()] for w in xs.squeeze()]
    #print(ngram)
    weights = torch.zeros(len(input_crop)).byte() # Batch version
    #print(window, N - S[1])
    weights[min(window, S[0]):-min(N-1-S[-1],window)] = 100
    #weights[-window:] = 0
    #print(weights)
    #pdb.set_trace()
    #print(S, contex_idx, len(contex_idx))
    #print(weights)
    #print(asd.asdas)

    #plot_text_explanation(ngram, 0*np.ones(5), ax = ax[0], n_cols = 5)
    plot_text_explanation(input_crop, weights, ax = ax[0], n_cols = 8)

    try:
        C_idxs = [np.where(V == k)[0][0] for k in C]
    except:
        pdb.set_trace()

    pc = hist[C_idxs].sum()
    # Make pc relative
    pc = pc / (hist.sum())

    ### Third pane shows entailed classes
    if sort_bars or (plot_type == 'treemap'): # treemap always need sorted vals
        arridx = hist.argsort()[::-1]
        hist = hist[arridx]
        classes = classes[arridx]

    if plot_type == 'bar':
        attrib_barplot(hist, C, classes, class_names=class_names, topk = topk, ax = ax[ncol-1], pc = pc)
    elif plot_type == 'treemap':
        if len(V) > 15:
            eff_topk = max(min(2*topk, len(C)), 10)
            labels = [class_names[a] for a in classes[:eff_topk]]
        else:
            labels = [class_names[a] for a in classes]
        treemap_boundary(hist, boundary = len(C), label = labels, ax= ax[ncol-1],
                         dynamic_fontsize=True, pc = pc)

    ax[ncol-1].set_title(r'$|V| = {}$, $|C| = {}$'.format(len(V),len(C)), fontsize = 14)
    #plt.tight_layout()
    if title:
        plt.suptitle(title, fontsize = 18)

def plot_2d_attrib_entailment(x, xs, S, C, V, hist, plot_type = 'bar',
              title = None, show_cropped = True,
              sort_bars = True, topk = 10, class_names = None, cmap = 'Greys',
              ax = None, save_path = None):
    """
        Plot of 2D input, masked attribute and entailed probs
    """

    if ax is None:
        ncol = 3 if show_cropped else 2
        fig, ax = plt.subplots(1, ncol, figsize = (4*ncol,4))
    else:
        ncol = len(ax)

    if type(V) is list:
        V = np.array(V)

    # Infer some aspects of input and mask
    nclass = len(hist)
    _, _, H, W   = x.shape
    classes = V #np.arange(nclass)
    S_x =  list(range(H)[S[0]])
    S_y =  list(range(W)[S[1]])
    ii, jj = S_x[0], S_y[0]
    h, w   = len(S_x), len(S_y)

    #ax[0].axis('off')
    im0 = ax[0].imshow(x.reshape(H, W),  aspect="auto", cmap = cmap)#, cmap='Greys_r')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_aspect('equal', 'box')
    #ax[0].set_title('Bla')
    if show_cropped:
        ax[1].imshow(xs.reshape(H, W), aspect="auto", cmap = cmap)
        #ax[1].axis('off')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_aspect('equal', 'box')
    for i in range(2):
        rect = patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax[i].add_patch(rect)

    try:
        C_idxs = [np.where(V == k)[0][0] for k in C]
    except:
        pdb.set_trace()

    pc = hist[C_idxs].sum()
    # Make pc relative
    pc = pc / (hist.sum())

    ### Third pane shows entailed classes
    if sort_bars or (plot_type == 'treemap'): # treemap always need sorted vals
        arridx = hist.argsort()[::-1]
        hist = hist[arridx]
        classes = classes[arridx]

    if plot_type == 'bar':
        attrib_barplot(hist, C, classes, class_names=class_names, topk = topk, ax = ax[ncol-1], pc = pc)
    elif plot_type == 'treemap':
        eff_topk = max(min(2*topk, len(C)), 10)
        labels = [class_names[a] for a in classes[:eff_topk]]
        treemap_boundary(hist, boundary = len(C), label = labels, ax= ax[ncol-1],
                         dynamic_fontsize=True, pc = pc)

    ax[ncol-1].set_title(r'$|V| = {}$, $|C| = {}$'.format(len(V),len(C)), fontsize = 14)

    if title:
        plt.suptitle(title, fontsize = 18)

def v_color(ob):
    COLOR = {
        True:  'red',
        False: '#ffcc33'
    }
    return COLOR[ob.is_simple]

def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)

def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)

def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color=v_color(ob), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

def treemap_boundary(sizes, boundary = None, norm_x = 100, norm_y = 100, topk = None,
                    value = None, dynamic_fontsize = False, label = None,
                    pc = None, ax = None, **kwargs):
    """
        Essentially, same as squarify.plot with the following changes:
         - custom (insetead of random) cmap
         - draw boundary
    """
    if boundary is not None:
        assert boundary in range(len(sizes))

    if ax is None:
        ax = plt.gca()


    cmap = matplotlib.cm.get_cmap('Set2')
    color = [cmap(i % cmap.N) for i in range(len(sizes))]
    normed = squarify.normalize_sizes(sizes, norm_x, norm_y)
    rects  = squarify.squarify(normed, 0, 0, norm_x, norm_y)

    x = [rect['x'] for rect in rects]
    y = [rect['y'] for rect in rects]
    dx = [rect['dx'] for rect in rects]
    dy = [rect['dy'] for rect in rects]

    br = ax.bar(x, dy, width=dx, bottom=y, color=color,
       label=label, align='edge', **kwargs)

    if boundary is not None:
        polygons = []
        for i,b in enumerate(br[:boundary]):
            #print(b)
            w,h = b.get_width(), b.get_height()
            x0, y0 = b.xy
            polygons.append(shapely.geometry.box(x0, y0, x0+w, y0+h))
        poly_union = shapely.ops.cascaded_union(polygons)
        plot_line(ax, poly_union.exterior)

    ax.axis('off')

    if not value is None:
        va = 'center' if label is None else 'top'

        for v, r in zip(value, rects):
            x, y, dx, dy = r['x'], r['y'], r['dx'], r['dy']
            ax.text(x + dx / 2, y + dy / 2, v, va=va, ha='center')

    if not label is None:
        va = 'center' if value is None else 'bottom'
        for l, r in zip(label, rects[:len(label)]):
            x, y, dx, dy = r['x'], r['y'], r['dx'], r['dy']
            fs = 0.9*min(dx,dy) if dynamic_fontsize else 10
            ax.text(x + dx / 2, y + dy / 2, l, va=va, ha='center', fontsize = fs)

    if pc is not None:
        ax.text(0, -5, r"$P(C)/P(V)$= %.2f" % pc, fontdict = {'color': 'red'})

    return ax

# From my SENN package
def plot_text_explanation(words, values, n_cols = 6, ax = None, save_path = None):
    import seaborn as sns
    # Get some pastel shades for the colors
    #colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    n_rows = int(min(len(values), len(words)) / n_cols) + 1

    # Plot bars and create text labels for the table
    if type(words) is str:
        words = words.split(' ')

    cellcolours = np.empty((n_rows, n_cols), dtype='object')
    celltext    = np.empty((n_rows, n_cols), dtype='object')

    for r in range(n_rows):
        for c in range(n_cols):
            idx = (r * n_cols + c)
            val =  values[idx] if (idx < len(values)) else 0
            cellcolours[r,c] = cmap(val)
            celltext[r,c] = words[idx] if (idx < len(words)) else ''

    if ax is None:
        fig, ax = plt.subplots()#figsize=(n_cols, n_rows))

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Add a table at the bottom of the axes
    tab = ax.table(cellText=celltext,
                          cellColours = cellcolours,
                          rowLabels=None,
                          rowColours=None,
                          colLabels=None,
                          colWidths = [.09]*n_cols,
                          cellLoc='center',
                          loc='center')
    tab.auto_set_font_size(False)

    for key, cell in tab.get_celld().items():
        cell.set_linewidth(0)

    tab.set_fontsize(8)
    tab.scale(1.5, 1.5)  # may help

    # Adjust layout to make room for the table:
    #plt.subplots_adjust(left=0.2, bottom=0.2)

    #plt.ylabel("Loss in ${0}'s".format(value_increment))
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('')
    ax.axis('off')
    ax.grid('off')
    if save_path:
        plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
    #plt.show()

from torch.utils.data.sampler import Sampler

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
