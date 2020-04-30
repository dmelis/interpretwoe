import os
import numpy as np
import math
import itertools
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

import pdb

try:
    from IPython.display import clear_output
except ImportError:
    pass # or set "debug" to something else or whatever

import shapely
import shapely.geometry #import MultiPolygon
import shapely.ops
import matplotlib
import squarify


def detect_kernel():
    try:
        __IPYTHON__
    except NameError:
        #print("Not in IPython Kernel")
        return 'terminal'
    else:
        #print("In IPython Kernel")
        return 'ipython'

KERNEL=detect_kernel()

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
    print(model_path, results_path)
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
def attrib_barplot(hist, C, classes, topk=10, cumhist=None, class_names=None, ax = None,
                   sort_bars = True, score = None, score_type = 'cumprob'):
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
    rotation = 90 if np.max(np.array([len(l) for l in ticklabs])) else 0
    bar = ax.bar(ticklabs,hist)
    # if cumhist is not None:
    #     print('here')
    #     print(cumhist)
    #     print(asd.asd)
        #ax.bar()
    ax.set_xticks(range(nbars))
    ax.set_xticklabels(ticklabs, rotation = rotation)
    if sort_bars and not (topk < len(hist)) and (len(C) <= topk):
        # Draw separation lineself.
        #This only makes sense if (i) histogram is shown sorted, |C| < topk
        ax.axvline(x=len(C)-0.5, c ='red')
        if score is not None:
            ttext =  r"P(C)/P(V)= %.2f" % score if score_type == 'cumprob' else r"woe(C:V\C ; X)= %.2f" % score
            ax.text(len(C), 0.75*ax.get_ylim()[1], ttext, fontdict = {'color': 'red'},
            #rotation=90,
            verticalalignment='center')#, transform = ax[ncol-1].transAxes)

def plot_attribute(x, block_idxs, title = None, ax = None, rgba = False, grey_out_rest=True):
    """
        x should be an image of size w x w.
        This assumes block_idxs are also squared.
    """
    if ax is None:
        fig, ax = plt.subplots()

    input_w = x.shape[0]
    block_w = int(np.sqrt(len(block_idxs)))

    ximg = x.clone()

    mask = torch.zeros(input_w*input_w, dtype=torch.uint8)
    mask[block_idxs] = torch.tensor(True)

    if grey_out_rest:
        if rgba:
            ximg[:,3].view(-1)[mask^1] *= 0.5
        else:
            ximg.view(-1)[mask^1] *= 0.05

    if rgba:
        ax.imshow(ximg.reshape(input_w,input_w, 4))
    else:
        ax.imshow(ximg.reshape(input_w,input_w), cmap = 'Greys', vmin = 0, vmax = 1)

    plt.axis('on')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    ii, jj = np.unravel_index(min(block_idxs), (input_w,input_w))
    w, h   = block_w, block_w
    rects = patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rects)
    if title:
        ax.set_title(title)
    #title = '{:2.2f}'.format(partial_woes[bidx])
    #ax.set_title('Woe ' + title + ' ({}/{})'.format(woe_ranks[bidx], nblocks))


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
    contex_idx = range(max(0,S[0] - window), 1+ min(N-1,S[-1]+window))
    input_crop = [vocab.itos[w.item()] for w in x.squeeze()[contex_idx]]
    if S[0] > window:
        input_crop[0] = '...'
    if S[1] < N - window:
        input_crop[-1] = '...'
    ngram = [vocab.itos[w.item()] for w in xs.squeeze()]
    weights = torch.zeros(len(input_crop)).byte() # Batch version
    weights[min(window, S[0]):-min(N-1-S[-1],window)] = 100
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
    return ax

def plot_2d_attrib_entailment(x, xs, S, C, V, hist, cumhist = None, woe = None, plot_type = 'bar',
              title = None, show_cropped = True,
              sort_bars = True, topk = 10, class_names = None, cmap = 'Greys',
              ax = None, ims = None, return_elements = False, save_path = None):
    """
        Plot of 2D input, masked attribute and entailed probs
    """

    if ax is None:
        ncol = 3 if show_cropped else 2
        fig, ax = plt.subplots(1, ncol, figsize = (4*ncol,4))
        #redraw = True
    else:
        ncol = len(ax)
        #redraw = len(ax[0].images) == 0


    # if timeout:
    #     print('here')
    #     def close_event():
    #         plt.close() #timer calls this function after 3 seconds and closes the window
    #     timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    #     timer.add_callback(close_event)
    #     timer.start()

    if type(V) in [list, tuple]:
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
    if ims is None:
        im0 = ax[0].imshow(x.reshape(H, W),  aspect="auto", cmap = cmap)#, cmap='Greys_r')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_aspect('equal', 'box')

    #ax[0].set_title('Bla')
    if show_cropped:
        if ims is None:
            im1 = ax[1].imshow(xs.reshape(H, W), aspect="auto", cmap = cmap)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_aspect('equal', 'box')
        else:
            ims[1].set_data(xs.reshape(H, W))

    rects = []
    for i in range(2):
        rects.append(patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none'))
        ax[i].add_patch(rects[i])


    try:
        C_idxs = [np.where(V == k)[0][0] for k in C]
    except:
        print('Failed C_idxs')
        pdb.set_trace()

    if woe is not None:
        hyp_score = woe
        score_type = 'woe'
    else:
        # Score will be P(C) / P(V)
        hyp_score = hist[C_idxs].sum() / (hist.sum())
        score_type = 'cumprob'

    ### Third pane shows entailed classes
    if sort_bars or (plot_type == 'treemap'): # treemap always need sorted vals
        #pdb.set_trace()
        arridx = hist.argsort()[::-1]
        hist = hist[arridx]
        classes = classes[arridx]

    if plot_type == 'bar':
        attrib_barplot(hist, C, classes, class_names=class_names, topk = topk, cumhist = cumhist,
        ax = ax[ncol-1], score = hyp_score, score_type = score_type)
    elif plot_type == 'treemap':
        eff_topk = max(min(2*topk, len(C)), 10)
        labels = [class_names[a] for a in classes[:eff_topk]]
        treemap_boundary(hist, boundary = len(C), label = labels, ax= ax[ncol-1],
                         dynamic_fontsize=True, score = hyp_score, score_type = score_type)

    ax[ncol-1].set_title(r'$|V| = {}$, $|C| = {}$'.format(len(V),len(C)), fontsize = 14)

    if title:
        plt.suptitle(title, fontsize = 18)
    if return_elements:
        return ax, rects, ims

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
                    score = None, score_type = 'cumprob', ax = None, **kwargs):
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

    if score and score_type == 'cumprob':
        ax.text(0, -5, r"$P(C)/P(V)$= %.2f" % score, fontdict = {'color': 'red'})
    elif score and score_type == 'woe':
        ax.text(0, -5, r"woe(C:V\C ; X)= %.2f" % score, fontdict = {'color': 'red'})

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
    ax.grid(False)
    if save_path:
        plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
    #plt.show()

def annotate_group(name, span, ax=None, orient='h', pad=None, shift=0.5):
    """Annotates a span of the x-axis (or y-axis if orient ='v')"""
    def annotate(ax, name, left, right, y, pad):
        xy = (left, y) if orient == 'h' else (y, left)
        xytext=(right, y-pad) if orient =='h' else (y+pad, right)
        valign = 'top' if orient =='h' else 'center'
        halign = 'center' if orient == 'h' else 'center'
        rot = 0 if orient == 'h' else 0
        if orient == 'h':
            connectionstyle='angle,angleB=90,angleA=0,rad=5'
        else:
            connectionstyle='angle,angleB=0,angleA=-90,rad=5'

        arrow = ax.annotate(name,
                xy=xy, xycoords='data',
                xytext=xytext, textcoords='data',
                annotation_clip=False, verticalalignment=valign,
                horizontalalignment=halign, linespacing=2.0,
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0,
                        connectionstyle=connectionstyle),
                fontsize=8, rotation=rot
                )
        return arrow
    if ax is None:
        ax = plt.gca()
    lim = ax.get_ylim()[0] if orient=='h' else ax.get_xlim()[1]
    min = lim + (shift if orient =='h' else shift)
    center = np.mean(span)
    #pad = 0.01 * np.ptp(lim) # I had this but seems to be always 0
    if pad is None:
        pad = 0.01 if orient == 'h' else 0.2
    left_arrow  = annotate(ax, name, span[0], center, min, pad)
    right_arrow = annotate(ax, name, span[1], center, min, pad)
    return left_arrow, right_arrow

def range_plot(X, x0=None, colnames = None, plottype = 'box', groups=None,
              x0_labels=None, color_values=True, ax=None):
    """
        If provided, groups should be an array of same rowlength of X, will be used as hue
    """
    plot_colors = {'pos': '#3C8ABE', 'neu': '#808080', 'neg': '#CF5246'}


    # This works only for binary, and assumes 0,1 in groups are neg/pos.
    # palette = sns.color_palette([plot_colors[v] for v in ['neg', 'pos']])

    #palette = sns.color_palette([plot_colors[v] for v in ['neg', 'pos']])
    palette = sns.color_palette('pastel')

    assert X.shape[1] == len(x0)

    X = X.copy()
    if x0 is not None:
        x0 = x0.copy()

    # if rescale:
    #     # We rescale so that boxplots are roughly aligned
    #     # (do so only for nonbinary fetaures)
    #     centers = np.median(X[:,self.nonbinary_feats], axis=0)
    #     X[:,self.nonbinary_feats] -= centers
    #
    #     scales = np.std(X[:,self.nonbinary_feats], axis=0)
    #     #scales = np.quantile(X[:,self.nonbinary_feats], .75, axis=0)
    #     X[:,self.nonbinary_feats[scales > 0]] /= scales[scales>0]
    #
    #     # Must also rescale x0, but display it's tr
    #     x[self.nonbinary_feats]   -= centers
    #     x[self.nonbinary_feats[scales > 0]] /= scales[scales>0]



    df = pd.DataFrame(X, columns = colnames)

    pargs = {}

    if groups is not None:
        #df = pd.concat([df, pd.DataFrame({'groups': groups})])
        df['groups'] = groups
        #
        # df = pd.DataFrame(np.hstack([X, self.Y[:,None].astype(int)]),
        #                   columns = list(self.features[feat_order]) + ['class'])
        # Will need to melt to plot splitting vy var
        #pdb.set_trace()
        df = df.melt(id_vars= ['groups'])
        pargs['hue'] = 'groups'
        pargs['x'] = 'value'
        pargs['y'] = 'variable'
    if not ax:
        fig, ax = plt.subplots(figsize=(8,8))
    if plottype == 'swarm':
        ax = sns.swarmplot(data=df, orient="h", palette="Set2", ax = ax, alpha = 0.5)
    else:
        ax = sns.boxplot(data=df, orient="h", **pargs, palette=palette, showfliers=False, ax = ax)#, boxprops=dict(alpha=.3))
        #ax.xaxis.set_label_text("")
        ax.set_xlabel(None)
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .2))
    if x0 is not None:
        line, = ax.plot(x0, range(X.shape[1]), 'kd', linestyle='dashed', ms=5, linewidth=0.3, zorder = 1000)
    if x0_labels is not None:
        xmin,xmax = ax.get_xlim()
        delta = xmax-xmin
        pad = 0.1*delta
        #farright = xmax - 0.5
        ax.set_xlim(xmin, xmax+pad) # Give some buffer to point labels
        for i,val in enumerate(x0_labels):
            #ax.text(x0[i]+0.5, i, txt, fontsize=10, zorder = 1000)
            if type(val) in [float, np.float64, np.float32]:
                if color_values:
                    cstr = 'neg' if (val <= -2) else ('pos' if val >=2 else 'neu')
                else:
                    cstr = 'neu'
                txt = '{:2.2e}'.format(val)
            else:
                cstr = 'neu'
                txt = '{:2}'.format(val)
            #txt = '{:2.2f}'.format(val) if type(val) is float else '{}'.format(val)
            #print(x0[i], val)
            ax.text(xmax+0.6*pad, i, txt, fontsize=10, zorder = 1001, ha='right',
                    color = plot_colors[cstr])
    if groups is not None:
        # Get rid of legend title
        handles, labels = ax.get_legend_handles_labels()
        ncol = 2
        if x0 is not None:
            # Also, add points to legend
            handles.append(line)
            labels.append('This example')
            ncol+=1
        ax.legend(handles=handles, labels=labels,loc='upper center',bbox_to_anchor=(0.5, -0.025),ncol=ncol)

    ax.set_title('Feature values of explained example vs training data')

    return ax
#     if return_ax:
#         return ax
#     else:
#         plt.show()
