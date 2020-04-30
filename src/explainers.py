import sys
import scipy as sp
import scipy.special
import numpy as np
import pandas as pd
from functools import partial
from attrdict import AttrDict
import itertools
import signal
import time
from tqdm import tqdm
import seaborn as sns
import colored

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
except ImportError:
    pass # or set "

import pdb


### Localß
from .utils import detect_kernel
from .utils import plot_2d_attrib_entailment, plot_text_attrib_entailment, plot_attribute
from .utils import annotate_group, range_plot
#from .methods import mnist_normalize, mnist_unnormalize
from .woe_utils import invlogit
from .scoring import normalized_deltas

KERNEL=detect_kernel()

# TODO: FIXME - this was almost working, but need to install jupyter widgets
# if KERNEL == 'ipython':
#     from tqdm import tqdm_notebook as tqdm
# else:
#     from tqdm import tqdm


class WoEExplanation(object):
    def __init__(self, pred_y, pred_prob, true_y, h_entailed, h_contrast,
                 base_lods, total_woe, attwoes, attrib_names, class_names):
        self.prnt_colors = {'pos': 'green',   'neu': 'grey_50', 'neg': 'red'}
        #self.prnt_colors = ['green', 'grey_50','red']
        self.plot_colors = {'pos': '#3C8ABE', 'neu': '#808080', 'neg': '#CF5246'}

        self.base_lods = base_lods
        self.attrib_names = attrib_names
        self.pred_y = pred_y
        self.true_y = true_y
        self.pred_prob = pred_prob
        self.attwoes = attwoes
        self.total_woe = total_woe
        self.h_entailed = h_entailed
        self.h_contrast = h_contrast
        self.class_names = class_names
        #self.τ = thresholds
        self.sorted_attwoes = [(i,x) for x,i in sorted(zip(attwoes,range(len(self.attrib_names))), reverse=True)]


        self.woe_thresholds = {
                  'Neutral': 1.15,
                  'Substantial': 2.3,
                  #'Strong': 3.35,
                  'Strong': 4.61,
                  'Decisive': np.inf}

        self.pos_colors = {'pos': '#3C8ABE', 'neu': '#808080', 'neg': '#CF5246'}


        # TODO: Maybe pass attwoes as dict, so we get names for free?

    def __repr__(self):
        self.print()
        return ""

    def print(self, header_only=False):#, idx, p, prob, y, y_num, y_den, prior, total_woe      sorted_partial_woes, threshold):
        #pname = self.class_names[p] # predicted class
        #yname = self.class_names[y] # true class
        #hname = self.class_names[y_num] # denominator in woe

        entailed_str = ",".join(["'" + self.class_names[c] + "'"  for c in self.h_entailed])
        contrast_str = ",".join(["'" + self.class_names[c] + "'" for c in self.h_contrast])
        print('Prediction: {} (p={:2.2f})'.format(self.class_names[self.pred_y], self.pred_prob))
        print('True class: {}'.format(self.class_names[self.true_y]))
        print('Bayes odds explanation:\n')
        print('   {:8.2f}     =        {:8.2f}  +  {:<8.2f}'.format(self.base_lods + self.total_woe, self.base_lods, self.total_woe))
        print('post. log-odds  =  prior log-odds  +  total_woe')
        if len(self.h_contrast) == len(self.class_names) - 1:
            print('\nTotal WoE in favor of {}: {:8.2f}'.format(entailed_str, self.total_woe))
        else:
            print("\nTotal WoE in favor of {} (against {}): {:8.2f}".format(
                        entailed_str, contrast_str, self.total_woe))

        if not header_only:
            colors = [self.prnt_colors[k] for k in ['pos', 'neu', 'neg']]
            ci = 0
            woesum = 0
            for i,woe in self.sorted_attwoes:
                #if np.abs(woe) < threshold: continue
                if ci == 0 and woe < self.τ: ci +=1
                if ci == 1 and woe < -self.τ: ci +=1
                print(colored.fg(colors[ci]) + 'woe({:>20}) = {:8.2f}'.format(self.attrib_names[i], woe))
                woesum += woe
            print(colored.attr('reset') + '     {:>20} = {:8.2f}'.format('sum', woesum))

    def plot(self, figsize=(8,4), ax = None, show = True, save_path = None):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)

        attrib_ord, woe = list(zip(*self.sorted_attwoes))
        woe = np.array([self.base_lods] + list(woe))
        categories = ['PRIOR LOG-ODDS'] + [self.attrib_names[i] for i in attrib_ord]
        y_pos = np.arange(len(self.attrib_names)+1)
        entailed_str = ','.join([self.class_names[c] for c in self.h_entailed])

        # Plot bars, color by singificance
        τ = self.woe_thresholds['Neutral']
        pos = woe >= τ
        neg = woe <= -τ
        neu  = (woe > -τ) & (woe < τ)


        v = np.fromiter(self.woe_thresholds.values(), dtype='float')
        vals = np.sort(np.concatenate([-v,v, np.array([0])]))
        vals = vals[(vals<10) & (vals>-10)]

        cmap = sns.color_palette("RdBu_r", len(self.woe_thresholds)*2)[::-1]#len(vals))

        bar_colors = [cmap[i] for i in np.digitize(woe, vals)]
        ax.barh(y_pos, woe, align='center', color = bar_colors)

        # if np.any(pos): ax.barh(y_pos[pos], woe[pos], align='center', color = self.plot_colors['pos'])
        # if np.any(neu):  ax.barh(y_pos[neu], woe[neu], align='center', color = self.plot_colors['neu'])
        # if np.any(neg): ax.barh(y_pos[neg], woe[neg], align='center', color = self.plot_colors['neg'])
        #




        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()  # labels read top-to-bottom
        #ax.set_xlabel('Weight-of-Evidence')
        ax.set_title("Weight of Evidence for Label: '"+entailed_str+"'", fontsize=14)
        ax.set_xlim(-1+min(np.min(self.attwoes),-6), max(6,np.max(self.attwoes))+1)
        ax.axhline(0.5, alpha=0.5, color='red', linestyle = ':')

        # Draw helper vertical lines delimiting woe regimes
        ax.axvline(0, alpha = 1, color = 'k', linestyle = '-')
        prev_τ = 0
        for level,τ in self.woe_thresholds.items():
            if -τ > ax.get_xlim()[0]:
                ax.axvline(-τ, alpha = 0.5, color = 'k', linestyle = '--')
            if τ < ax.get_xlim()[1]:
                ax.axvline(τ, alpha = 0.5, color = 'k', linestyle = '--')

            if level == 'Neutral':
                annotate_group('Not\nSignificant.', (-τ,τ), ax)
            else:
                annotate_group('{}\nIn Favor'.format(level), (prev_τ,min(τ, ax.get_xlim()[1])), ax)
                annotate_group('{}\nAgainst'.format(level), (max(-τ, ax.get_xlim()[0]),-prev_τ), ax)

            #annotate_group('Significant\nAgainst', (ax.get_xlim()[0],-τ), ax)
            prev_τ = τ
        #ax.text(-2.5,0,'significative against',horizontalalignment='right')

        print()



        if save_path:
            plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
        if show:
            plt.show()

        return ax

class Explanation(object):
    def __init__(self, x, y, classes, masker = None, task = None, **kwargs):
        self.input = x
        self.prediction = y
        self.classes = classes
        self.predicates = []
        self.masker = masker # Must be passed when instantiating Explanation
        self.task   = task   # Must be passed when instantiating Explanation
        # For text tasks
        self.vocab = kwargs['vocab'] if 'vocab' in kwargs else None

    def plot(self, plot_type = 'treemap', mask_alpha = 0, save_path = None):
        """
            mask_mode: 'grey' or 'white'
        """
        n = len(self.predicates)
        if self.task in ['ets']:
            fig, ax = plt.subplots(n, 2, figsize = (12,4*n),
                                        gridspec_kw = {'width_ratios':[2, 1]})
        else:
            ncol = 3
            fig, ax = plt.subplots(n, 3, figsize =  (4*ncol,4*n),
                                        gridspec_kw = {'width_ratios':[1, 1, 1.3]})

        #V = np.array(list(range(len(self.classes))))

        C_prev = range(len(self.classes))
        for i, pred in enumerate(self.predicates):
            ax[i,0].set_title('Most informative attrib (Step {})'.format(i))
            #S,C,hist_V, V_prev, V =
            #locals().update(dict(pred)) # loads pred keys to namespace
            row_ax = ax[i,:] if len(self.predicates) >1 else ax
            masked = self.masker(self.input, pred.S, alpha = mask_alpha)

            if type(masked) is tuple:
                X_s, X_s_plot = masked
            else:
                X_s = masked

            if self.task in ['hasy', 'mnist','leafsnap']:
                plot_2d_attrib_entailment(
                    self.input, X_s_plot, pred.S, pred.hyp, pred.V_prev, pred.hist,
                    woe= pred.total_woe if 'total_woe' in pred else None,
                    plot_type=plot_type, cumhist = pred.cumhist,
                    class_names = self.classes, topk = 10, #title = labstr,
                    sort_bars = True, ax = row_ax, save_path = save_path)
            elif self.task in ['ets']:
                plot_text_attrib_entailment(self.input, X_s, pred.S, pred.hyp, pred.V_prev, pred.hist,
                                    woe = pred.woe if 'woe' in pred else None,
                                    plot_type = plot_type,
                                    class_names = self.classes,
                                    vocab = self.vocab, ax = row_ax,
                                    title = None, sort_bars = False)
        #if self.task in ['ets']:
        plt.suptitle('Prediction: {} (class {})'.format(
                            self.classes[self.prediction], self.prediction),
                            fontsize = 24, y = .95)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
        print('Done!')


class Explainer(object):
    """
        Parent class for explainers.

    """
    def __init__(self, classifier, classes, features, task, X, Y, input_size,  loss_type, reg_type, alpha, p, plot_type):
        self.task = task
        self.X, self.Y = X,Y # usually training data, for plotting / viz purposes
        self.classifier = classifier
        self.classes    = classes
        self.features   = features
        self.plot_type  = plot_type
        self.loss_type  = loss_type
        self.reg_type   = reg_type
        self.alpha      = alpha
        self.p          = p
        self.task       = task
        self.input_size = input_size

        # Detect binary features - will be used later in plotting etc
        binary_mask = np.sum(np.round(X) == X, axis=0) == len(X)
        self.binary_feats = np.where(binary_mask)[0]
        self.nonbinary_feats = np.where(np.logical_not(binary_mask))[0]

    def _init_criterion(self):

        loss = self.loss_type

        if loss == 'norm_delta':
            crit = partial(normalized_deltas, reg_type = self.reg_type,
                    alpha=self.alpha, p=self.p, argmin_reg = None,
                    include_pred= True)
        else:
            raise ValueError('Not implemented yet')

        return crit

    def explain(self):
        pass



class WOE_Explainer(Explainer):
    """
        Multi-step Contrastive Explainer.


        - alpha, p are parameters for the explanation scoring function - see their description there
    """
    def __init__(self, classifier, woe_model, X=None, Y=None, classes=None, features=None,
                featgroup_idxs=None,
                featgroup_names = None,
                task='tabular',
                input_size=None,  loss_type = 'norm_delta',
                reg_type = 'decay', alpha=1, p=1, plot_type = 'bar'):
        super().__init__(classifier, classes, features, task, X, Y, input_size, loss_type, reg_type, alpha, p, plot_type)

        self.woe_model  = woe_model
        self.featgroup_idxs = featgroup_idxs
        self.featgroup_names = featgroup_names


        self.woe_thresholds = {
                  'Neutral': 1.15,
                  'Substantial': 2.3,
                  #'Strong': 3.35,
                  'Strong': 4.61,
                  'Decisive': np.inf}

        # TODO: maybe combine into dict, but we need to maintain order of keys!
        #self.attributes = attributes # dict with 'attrib_name': [indices] pairs


        # self.plot_type  = plot_type
        # self.loss_type  = loss_type
        # self.reg_type   = reg_type
        #self.alpha      = alpha
        #self.p          = p
        # self.task       = woe_model.task
        # self.input_size = woe_model.input_size
        #self.input_size = woe_model.image_size # Legacy. Replace by above once new models trainerd.
        #self.pykernel   = detect_kernel()

        if self.task == 'mnist':
            H, W = self.input_size
            self.masker = partial(image_masker, normalized = False, H=H, W=W)
            self.input_type = 'image'
        elif self.task in ['hasy', 'leafsnap']:
            H, W = self.input_size
            self.masker = partial(image_masker_unnormalized, H=H, W=W)# #partial(hasy_masker)#,  h = model.mask_size, w = model.mask_size)    )
            self.input_type = 'image'
        elif self.task == 'ets':
            self.masker = ets_masker
            self.input_type = 'text'
        elif self.task == 'tabular':
            self.masker = None
            self.input_type = 'tabular'
        else:
            raise ValueError('Unrecognized task.')
        self.criterion = self._init_criterion()


    def _get_explanation(self, x, y_pred, y_prob, y_true, hyp, null_hyp, units='groups'):
        """ Base function to compute explanation - generic, can take simple or complex hyps
            but a single example at a time.
        """
        assert x.ndim == 1
        assert type(hyp) in [np.ndarray, list]
        assert type(null_hyp) in [np.ndarray, list]

        ### Total WOE and Prior Odds
        total_woe   = self.woe_model.woe(x, hyp, null_hyp)
        prior_lodds = self.woe_model.prior_lodds(hyp, null_hyp)

        if 'group' in units:
            ### Compute Per-Attribute WoE Scores
            woes = []
            for i,idxs in enumerate(self.featgroup_idxs):
                woes.append(self.woe_model.woe(x, hyp, null_hyp, subset = idxs))
            woes = np.array(woes).T
            woe_names = self.featgroup_names
            #sorted_attrwoes= [(i,x) for x,i in sorted(zip(attrwoes[idx],range(len(self.attribute_names))), reverse=True)]
        elif units == 'features':
            ### Compute also per-feature woe for plot
            woes = []
            for i in range(x.shape[0]):
                woes.append(self.woe_model.woe(x,  hyp, null_hyp, subset = [i]))
            woes = np.array(woes).T
            woe_names = self.features

        ### Create Explanation
        #hyp_classes  = [self.classes[c] for c in hyp]
        #null_classes = [self.classes[c] for c in null_hyp]
        expl = WoEExplanation(y_pred, y_prob, y_true,
                              hyp, null_hyp,
                              prior_lodds, total_woe, woes,
                              woe_names, self.classes)

        return expl
        # plot = True
        # show_ranges = False
        # if not plot: return expl
        # ncol = 2 if show_ranges else 1
        # fig, ax = plt.subplots(1,ncol, figsize=(6*ncol, 8))
        # ax0 = ax[0] if show_ranges else ax
        # ax0 = expl.plot(ax=ax0, show=False)
        # if show_ranges:
        #     # Sort Features by order of their attribute in first plot
        #     feat_order = np.hstack([self.attribute_idxs[i] for i,_ in expl.sorted_attwoes])
        #
        #     if range_group == 'entailed':
        #         mask = self.Y == y_num # show ranges for the entailed hyp
        #         groupby = None
        #     elif range_group == 'rejected':
        #         mask = self.Y == y_den # show ranges for rejected hyp
        #         groupby = None
        #     elif range_group == 'split':
        #         mask = range(self.X.shape[0]) # no filter
        #         groupby = [self.classes[i] for i in self.Y]
        #
        #     #pnt_labels = ['{:2.2f}'.format(w) for w in featwoes[idx][feat_order]]
        #     pnt_labels = featwoes[idx][feat_order]
        #     ax1 = range_plot(self.X[mask][:,feat_order], x[idx, feat_order],
        #                      colnames = self.features[feat_order], groups=groupby,
        #                      x0_labels=pnt_labels, plottype='box', ax = ax[1])
        #     ax1.set_title('Feature values of explained example vs {} class'.format(range_group))
        #
        #     offset = -0.5
        #     #for i in range(1,len(self.attribute_names)):
        #     for i in np.argsort(attrwoes[0])[::-1]:
        #         new_offset = offset + len(self.attribute_idxs[i])
        #         ax1.axhline(new_offset, alpha=0.5, color='gray', linestyle = ':')
        #         annotate_group(self.attribute_names[i].replace(' ','\n'), (offset, new_offset), ax1, orient='v')
        #         offset = new_offset
        #
        # plt.tight_layout()
        # if save_path:
        #         plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
        # plt.show()


    # def seqexplain(self, x, y, idx = 0, threshold = 2, show_ranges=True, range_group='split',
    #             totext=True, favor_class = None, plot=True, save_path = None):
    #     # TODO: Generalize. Can we present explanations for all elements in batch
    #     # like interpml does? Instead of having to choose idx.
    #     #pred_class = self.classifier(x)
    #     idx = 0
    #     pred_class = self.classifier.predict(x)
    #     pred_proba = self.classifier.predict_proba(x)
    #
    #     V = list(range(len(self.classes)))
    #     E = Explanation(x, y, self.classes, masker = self.masker, task = self.task)
    #     step = 0
    #
    #     self.woe_model._start_caching()
    #     while len(V) > 1:
    #         print('\n' + ''.join(20*['-']) + '\nExplanation step: ', step)
    #
    #         if verbose > 1:
    #             print(V)
    #         ## First, optimize over hypothesis partitions
    #         try:
    #             hyp, null_hyp, hyp_woe = self.choose_hypothesis(x, y, V)
    #             print('Contrasting hypotheses: {}/{}'.format(hyp, null_hyp))
    #         except:
    #             pdb.set_trace()
    #             hyp, null_hyp, hyp_woe = self.choose_hypothesis(x, y, V)
    #
    #         #self.explain(x, )
    #         ### Total WOE and Prior Odds
    #         total_woe  = self.woe_model.woe(x, hyp, null_hyp)
    #         prior_lodds = self.woe_model.prior_lodds(hyp, null_hyp)
    #
    #         ### Compute Per-Attribute WoE Scores
    #         attrwoes = []
    #         for i,idxs in enumerate(self.attribute_idxs):
    #             attrwoes.append(self.woe_model.woe(x, y_num, y_den, subset = idxs))
    #         attrwoes = np.array(attrwoes).T
    #         #sorted_attrwoes= [(i,x) for x,i in sorted(zip(attrwoes[idx],range(len(self.attribute_names))), reverse=True)]
    #
    #         ### NEW: Compute also per-feature woe for plot
    #         featwoes = []
    #         for i in range(x.shape[1]):
    #             featwoes.append(self.woe_model.woe(x, y_num, y_den, subset = [i]))
    #         featwoes = np.array(featwoes).T
    #
    #         ### Create Explanation
    #         print('Example index: {}'.format(idx))
    #         expl = WoEExplanation(self.classes[pred_class[idx]], pred_proba[idx,pred_class[idx]],
    #                               self.classes[y[idx]], self.classes[y_num], self.classes[y_den],
    #                               prior_lodds, total_woe[idx], attrwoes[idx],
    #                               self.attribute_names)
    #
    #         expl.print(header_only = not totext)
    #
    #         featexpl =  WoEExplanation(self.classes[pred_class[idx]], pred_proba[idx,pred_class[idx]],
    #                               self.classes[y[idx]], self.classes[y_num], self.classes[y_den],
    #                               prior_lodds, total_woe[idx], featwoes[idx],
    #                               self.features)
    #
    #
    #         if not plot: return expl
    #
    #         ncol = 2 if show_ranges else 1
    #         fig, ax = plt.subplots(1,ncol, figsize=(6*ncol, 8))
    #         ax0 = ax[0] if show_ranges else ax
    #         ax0 = expl.plot(ax=ax0, show=False)
    #         if show_ranges:
    #             # Sort Features by order of their attribute in first plot
    #             feat_order = np.hstack([self.attribute_idxs[i] for i,_ in expl.sorted_attwoes])
    #
    #             if range_group == 'entailed':
    #                 mask = self.Y == y_num # show ranges for the entailed hyp
    #                 groupby = None
    #             elif range_group == 'rejected':
    #                 mask = self.Y == y_den # show ranges for rejected hyp
    #                 groupby = None
    #             elif range_group == 'split':
    #                 mask = range(self.X.shape[0]) # no filter
    #                 groupby = [self.classes[i] for i in self.Y]
    #
    #             #pnt_labels = ['{:2.2f}'.format(w) for w in featwoes[idx][feat_order]]
    #             pnt_labels = featwoes[idx][feat_order]
    #             ax1 = range_plot(self.X[mask][:,feat_order], x[idx, feat_order],
    #                              colnames = self.features[feat_order], groups=groupby,
    #                              x0_labels=pnt_labels, plottype='box', ax = ax[1])
    #             ax1.set_title('Feature values of explained example vs {} class'.format(range_group))
    #
    #             offset = -0.5
    #             #for i in range(1,len(self.attribute_names)):
    #             for i in np.argsort(attrwoes[0])[::-1]:
    #                 new_offset = offset + len(self.attribute_idxs[i])
    #                 ax1.axhline(new_offset, alpha=0.5, color='gray', linestyle = ':')
    #                 annotate_group(self.attribute_names[i].replace(' ','\n'), (offset, new_offset), ax1, orient='v')
    #                 offset = new_offset
    #
    #         plt.tight_layout()
    #         if save_path:
    #                 plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
    #         plt.show()
    #     #featexpl.print()
    #     return expl

    def explain(self, x, y, idx = 0, threshold = 2, sequential=False,
                show_ranges=True, range_group='split', units='attributes',
                totext=True, favor_class = 'predicted', plot=True, verbose=False, save_path = None):
        # TODO: Generalize. Can we present explanations for all elements in batch
        # like interpml does? Instead of having to choose idx.
        #pred_class = self.classifier(x)
        idx = 0
        y_preds = self.classifier.predict(x)
        y_probs = self.classifier.predict_proba(x)

        print('Example index: {}'.format(idx))
        x = x[idx]
        y = y[idx]
        y_pred = y_preds[idx]
        y_prob = y_probs[idx, y_preds].item()

        # If not sequential, this is the only y in numerator. If sequential,
        # this is the class that must be contained in all denominators
        if favor_class is not None and type(favor_class) is int:
            y_num = favor_class
        elif favor_class == 'predicted':
            y_num = int(y_pred)
        elif favor_class == 'true':
            y_num = int(y)
        else:
            y_num = 0


        # TODO: Generalize to multiclass
        if not sequential:

            y_num = [y_num]
            y_den = sorted(list(set(range(len(self.classes))) - set(y_num)))
            #y_den = int( not bool(y_num)) # works only for binary

            ### Create Explanation
            #pdb.set_trace()
            expl = self._get_explanation(x, y_pred, y_prob, y, hyp=y_num, null_hyp=y_den, units=units)

            expl.print(header_only = not totext)

            featexpl = self._get_explanation(x, y_pred, y_prob, y, hyp=y_num, null_hyp=y_den, units='features')

            if plot:
                ncol = 2 if show_ranges else 1
                figsize = (12,8) if show_ranges else (12,8)
                fig, ax = plt.subplots(1,ncol, figsize=figsize)
                ax0 = ax[0] if show_ranges else ax
                ax0 = expl.plot(ax=ax0, show=False)
                if show_ranges:
                    self.plot_ranges(x, expl, featexpl, range_group, ax = ax[1])
                plt.tight_layout()
                if save_path:
                        plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
                plt.show()
        else:

            V = list(range(len(self.classes)))
            #E = Explanation(x, y, self.classes, masker = self.masker, task = self.task)
            step = 1
            #self.woe_model._start_caching()
            while len(V) > 1:
                print('\n' + ''.join(20*['-']) + '\nExplanation step: ', step)

                if verbose > 1:
                    print(V)
                ## First, optimize over hypothesis partitions
                try:
                    hyp, null_hyp, hyp_woe = self.choose_hypothesis(x, y_num, V)
                    print('Contrasting hypotheses: {}/{}'.format(hyp, null_hyp))
                except:
                    pdb.set_trace()
                    hyp, null_hyp, hyp_woe = self.choose_hypothesis(x, y_num, V)

                #pdb.set_trace()
                expl = self._get_explanation(x,y_pred, y_prob, y, hyp=hyp, null_hyp=null_hyp, units=units)

                expl.print(header_only = not totext)

                featexpl = self._get_explanation(x, y_pred, y_prob, y, hyp=hyp, null_hyp=null_hyp, units='features')

                if plot:
                    ncol = 2 if show_ranges else 1
                    figsize = (12,8) if show_ranges else (12,8)
                    fig, ax = plt.subplots(1,ncol, figsize=figsize)
                    ax0 = ax[0] if show_ranges else ax
                    ax0 = expl.plot(ax=ax0, show=False)
                    if show_ranges:
                        self.plot_ranges(x, expl, featexpl, range_group, ax = ax[1])
                    plt.tight_layout()
                    if save_path:
                            plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
                    plt.show()

                ## Remove classes from reference set
                V_prev = V.copy()
                V = [k for k in V if k in hyp]  # np version: np.isin(V, max_C)

                if verbose > 0:
                    print('Shrinking ground set: {} -> {}'.format(V_prev, V))
                step += 1


        #pdb.set_trace()


        # ### Total WOE and Prior Odds
        # total_woe  = self.woe_model.woe(x, y_num, y_den)
        # prior_odds = self.woe_model.prior_odds(y_num, y_den)
        #
        # ### Compute Per-Attribute WoE Scores
        # attrwoes = []
        # for i,idxs in enumerate(self.attribute_idxs):
        #     attrwoes.append(self.woe_model.woe(x, y_num, y_den, subset = idxs))
        # attrwoes = np.array(attrwoes).T
        # #sorted_attrwoes= [(i,x) for x,i in sorted(zip(attrwoes[idx],range(len(self.attribute_names))), reverse=True)]
        #
        # ### NEW: Compute also per-feature woe for plot
        # featwoes = []
        # for i in range(x.shape[1]):
        #     featwoes.append(self.woe_model.woe(x, y_num, y_den, subset = [i]))
        # featwoes = np.array(featwoes).T

        ### Print terminal-friendly explanation - before I created the WoeExplanation class
        # self.print_explanation(idx, pred_class[idx], pred_proba[idx,pred_class[idx]], y[idx],
        #                        y_num,y_den, prior_odds, total_woe[idx],
        #                        sorted_partial_woes, threshold)

        # ### Compare value of most informative attrib to mean values per class
        # if hasattr(self.classifier, 'theta_'):
        # # Maybe instead make class means a property of the woe model, change to:
        # #if hasattr(self.woe_model, 'class_means')
        #     top_attrib      = np.argmax(np.abs(attrwoes), axis=1)[idx]
        #     top_attrib_idxs = self.attribute_idxs[top_attrib]
        #     vals_top_attrib = x[0,top_attrib_idxs]
        #     top_attrib_name = self.attribute_names[top_attrib]
        #     print('\nAverage "{}" value for {} examples =\n\t{}'.format(top_attrib_name, self.classes[y_num],self.classifier.theta_[y_num, top_attrib_idxs]))
        #     print('Average "{}" value for {} examples =\n\t{}'.format(top_attrib_name, self.classes[y_den],self.classifier.theta_[y_den, top_attrib_idxs]))
        #     print('Value of "{}" in this example =\n\t{}'.format(top_attrib_name, vals_top_attrib))


        # featexpl =  WoEExplanation(self.classes[pred_class[idx]], pred_proba[idx,pred_class[idx]],
        #                       self.classes[y[idx]], self.classes[y_num], self.classes[y_den],
        #                       prior_odds, total_woe[idx], featwoes[idx],
        #                       self.features)




        #featexpl.print()
        return expl




    def plot_priors(self, ax = None, normalize = 'chance'):
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        else:
            show = False
        probs = self.woe_model.class_prior
        max_prob = np.round(probs.max()+0.1, 1)
        prob_tickvals = np.around(np.arange(0.05,max_prob, 0.05), decimals=2)
        #prob_tickvals = np.around(np.linspace(0,np.round(probs.max()+0.1,1),10, endpoint=False), decimals=2)

        if normalize is None:
            # These two shoudl be equivalent:
            #priors = np.array([self.woe_model.prior_lodds(i) for i in range(len(self.classes))])
            priors = sp.special.logit(probs)
            assert np.allclose(sp.special.expit(priors).sum(), 1)
            label = r'$\log \frac{p(y)}{1-p(y)}$'
            prob_tickvals_transform = sp.special.logit(prob_tickvals)
        elif normalize == 'max':
            priors = np.log(self.woe_model.class_prior/self.woe_model.class_prior.max())
            label = r'$\log \frac{p(y)}{\max_{y\'} p(y\')}$'
        elif normalize == 'chance':
            priors = np.log(self.woe_model.class_prior/(1/len(self.classes)))
            label = r'$\log \frac{p(y)}{' +'1/{}'.format(len(self.classes))+ '}$'#.format(1/len(self.classes))
            prob_tickvals_transform = np.log(prob_tickvals/(1/len(self.classes)))
        else:
            raise ValueError()


        assert np.allclose(probs.sum(), 1)

        #print(probs.sum())
        #print(sp.special.expit(priors).sum())
        ticklabelpad = mpl.rcParams['xtick.major.pad']

        label_fs = 14

        v = np.fromiter(self.woe_thresholds.values(), dtype='float')
        vals = np.sort(np.concatenate([-v,v, np.array([0])]))
        vals = vals[(vals<10) & (vals>-10)]
        cmap = sns.color_palette("RdBu_r", len(self.woe_thresholds)*2)[::-1]#len(vals))
        bar_colors = [cmap[i] for i in np.digitize(priors, vals)]
        ax.barh(range(len(self.classes)), priors, align='center', color = bar_colors)
        ax.set_yticks(range(len(self.classes)))
        ax.set_yticklabels(self.classes)
        #ax.set_xlabel(label)
        ax.annotate(label, xy=(1,0), xytext=(5, -ticklabelpad), ha='left', va='center',
            xycoords='axes fraction', textcoords='offset points',fontsize=label_fs)

        if normalize == 'chance':
            ax.axvline(0, alpha = 1, color = 'gray', linestyle = '--')
            annotate_group('Less than Chance', (ax.get_xlim()[0], 0), ax, shift=-.75)
            annotate_group('More than Chance', (0,ax.get_xlim()[1]), ax, shift=-.75)
            ax.set_title('Class Prior Log Odds (vs chance)')
        elif normalize is None:
            ax.set_title('Class Prior Log Odds (vs all other classes)')

        # Twin axis
        ax2 = ax.twiny()
        ax2.set_xlabel('')
        ax2.annotate(r'p(y)', xy=(1,1), xytext=(5, -ticklabelpad), ha='left', va='center',
            xycoords='axes fraction', textcoords='offset points', fontsize=label_fs)
        ax2.set_xticks(prob_tickvals_transform)
        ax2.set_xticklabels(prob_tickvals)
        ax2.set_xlim(ax.get_xlim())
        if show:
            plt.show()

    def plot_ranges(self, _x, expl=None, featexpl=None, groupby = 'entailed',
                    annotate='woe', rescale= False, ax = None):
        # TODO: Make this a class method of WoEExplanation
        # The later might require refactoring things so that all explanations
        # have both att and feat woes. All that would change in _get_explanation,
        # here we just call _get_explanation once.
        # UPDATE: Maybe keep it here, so that we can use it with our without explanations

        """
                groupy: how the train data should be grouped to show boxplots
                annotate: woe or value
        """
        x = _x.copy()
        if x.ndim > 1: x = x.squeeze()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,20))

        if expl is not None:
            # Sort Features by order of their attribute in first plot
            group_order = np.argsort(expl.attwoes)[::-1]
            feat_order = np.hstack([self.featgroup_idxs[i] for i,_ in expl.sorted_attwoes])
        else:
            # Sort by feature group order
            group_order = range(len(self.featgroup_idxs))
            feat_order = np.concatenate(self.featgroup_idxs)

        if groupby == 'split':
            mask = range(self.X.shape[0]) # no filter
            _groupby = [self.classes[i] for i in self.Y]
        elif groupby == 'predicted':
            mask = range(self.X.shape[0]) # no filter
            pred = self.classes[self.classifier.predict(x.reshape(1,-1))[0]]
            _groupby = [pred if self.classes[i]==pred else 'other classes' for i in self.Y]
        elif groupby == 'entailed':
            #mask = self.Y == y_num # show ranges for the entailed hyp - works only for binary
            mask = np.isin(self.Y, expl.h_entailed)
            _groupby = None
        elif groupby == 'rejected':
            mask = np.isin(self.Y, expl.h_contrast) # show ranges for rejected hyp
            _groupby = None

        #pnt_labelsr = ['{:2.2f}'.format(w) for w in featwoes[idx][feat_order]]
        if featexpl is None or annotate == 'value':
            pnt_labels = x[feat_order]
            color_vals = False
        elif annotate == 'woe':
            pnt_labels = featexpl.attwoes[feat_order]
            color_vals = True
        else:
            raise ValueError()

        X = self.X[mask].copy()
        if rescale:
            # We rescale so that boxplots are roughly aligned
            # (do so only for nonbinary fetaures)
            centers = np.median(X[:,self.nonbinary_feats], axis=0)
            X[:,self.nonbinary_feats] -= centers
            x[self.nonbinary_feats]   -= centers

            scales = np.std(X[:,self.nonbinary_feats], axis=0)
            #scales = np.quantile(X[:,self.nonbinary_feats], .75, axis=0)
            X[:,self.nonbinary_feats[scales > 0]] /= scales[scales>0]
            x[self.nonbinary_feats[scales > 0]] /= scales[scales>0]

        ax1 = range_plot(X[:,feat_order], x[feat_order],
                         colnames = self.features[feat_order], groups=_groupby,
                         x0_labels=pnt_labels, plottype='box', color_values=color_vals,
                         ax = ax)
        ax1.set_title('Feature values of this example compared to training data')#relative to other {} class'.format(groupby))

        offset = -0.5
        #for i in range(1,len(self.attribute_names)):
        for i in group_order:
            new_offset = offset + len(self.featgroup_idxs[i])
            ax1.axhline(new_offset, alpha=0.5, color='gray', linestyle = ':')
            annotate_group(self.featgroup_names[i].replace(' ','\n'), (offset, new_offset), ax, orient='v')
            offset = new_offset

        if rescale:
            # Don't show bvalues on xaxis
            ax1.set_xticks([])

        return ax


    def plot_explanation(self, sorted_partial_woes, figsize = (4,3)):
        # DEPRECATED IN FAVOR OF plot METHOD IN WoEExplanation??
        fig, ax = plt.subplots(figsize=figsize)
        color_pos, color_neu, color_neg = '#3C8ABE', '#808080','#CF5246'
        attrib_ord, woe = list(zip(*sorted_partial_woes))
        woe = np.array([base_logodds] + list(woe))
        categories = ['PRIOR LOG-ODDS'] + [attribute_names[i] for i in attrib_ord]
        y_pos = np.arange(len(attribute_names)+1)

        positive = woe >= threshold
        negative = woe <= -threshold
        neutral  = (woe > -threshold) & (woe < threshold)
        if np.any(positive): ax.barh(y_pos[positive], woe[positive], align='center', color = color_pos)
        if np.any(neutral): ax.barh(y_pos[neutral], woe[neutral], align='center', color = color_neu)
        if np.any(negative): ax.barh(y_pos[negative], woe[negative], align='center', color = color_neg)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()  # labels read top-to-bottom
        #ax.set_xlabel('Weight-of-Evidence')
        ax.set_title('Per-Attribute Weight of Evidence')
        ax.set_xlim(-1+min(np.min(partial_woes),-6), max(6,np.max(partial_woes))+1)
        ax.axvline(2, alpha = 0.5, color = 'k', linestyle = '--')
        ax.axvline(0, alpha = 1, color = 'k', linestyle = '-')
        ax.axvline(-2, alpha = 0.5, color = 'k', linestyle = '--')
        ax.axhline(0.5, alpha=0.5, color='red', linestyle = ':')
        #ax.text(-2.5,0,'significative against',horizontalalignment='right')

        print()

        annotate_group('Significant\nAgainst', (ax.get_xlim()[0],-threshold), ax)
        annotate_group('Neutral\nLow Sign.', (-threshold, threshold), ax)
        annotate_group('Significant\nIn Favor', (threshold,ax.get_xlim()[1]), ax)
        if save_path:
            plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
        plt.show()


    def print_explanation(self, idx, p, prob, y, y_num, y_den, prior, total_woe,
                          sorted_partial_woes, threshold):
        pname = self.classes[p] # predicted class
        yname = self.classes[y] # true class
        hname = self.classes[y_num] # denominator in woe
        print('Example index: {}'.format(idx))
        print('Prediction: {} (p={:2.2f})'.format(pname, prob))
        print('True class: {}'.format(yname))
        print('Bayes odds explanation:\n')
        print('   {:8.2f}     =        {:8.2f}  +  {:<8.2f}'.format(prior + total_woe, prior, total_woe))
        print('post. log-odds  =  prior log-odds  +  total_woe')
        print('\nTotal WoE in favor of "{}": {:8.2f}'.format(hname, total_woe))
        color_pos, color_neu, color_neg = 'green', 'grey_50','red'
        colors = [color_pos,color_neu,color_neg]
        ci = 0
        woesum = 0
        for i,woe in sorted_partial_woes:
            #if np.abs(woe) < threshold: continue
            if ci == 0 and woe < threshold: ci +=1
            if ci == 1 and woe < -threshold: ci +=1
            print(colored.fg(colors[ci]) + 'woe({:>20}) = {:8.2f}'.format(self.featgroup_names[i], woe))
            woesum += woe
        print(colored.attr('reset') + '     {:>20} = {:8.2f}'.format('sum', woesum))




    #@torch.no_grad()  ##FIXME - how to do this without global torch import
    def torchexplain(self, x, y = None, show_plot = 1, max_attrib = 5, save_path = None, verbose = False):
        # if y not provided, call classif model
        if y is None:
            out = self.classifier(x)
            p, y = out.max(1)
            y = y.item()

        if show_plot > 1 and self.input_type == 'image':
            fig = plt.figure( figsize = (4,4))
            plt.imshow(invlogit(x).reshape(28,28), cmap='Greys')
            plt.title('Prediction: ' + self.classes[y] + ' (true class no.{})'.format(y), fontsize = 20)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        else:
            print(y)
            print(self.classes)
            print('Prediction: ' + self.classes[y] + ' (true class: no.{})'.format(y))

        V = list(range(len(self.classes)))
        E = Explanation(x, y, self.classes, masker = self.masker, task = self.task)
        step = 0

        print('\nNOTE: For each step, we display only attributes with WOE > 5 (Positive Evidence)')
        print('or < -5 (Negative Evidence, if any such exist), up to at most five attributes per step.')

        self.woe_model._start_caching()
        while len(V) > 1:
            print('\n' + ''.join(20*['-']) + '\nExplanation step: ', step)

            if verbose > 1:
                print(V)
            ## First, optimize over hypothesis partitions
            try:
                hyp, null_hyp, hyp_woe = self.choose_hypothesis(x, y, V)
                print('Contrasting hypotheses: {}/{}'.format(hyp, null_hyp))
            except:
                pdb.set_trace()
                hyp, null_hyp, hyp_woe = self.choose_hypothesis(x, y, V)

            ## Do WOE decomposition for chosen hypothesis
            total_woe, partial_woes, attrib_idxs = self.woe_model.decomposition_woe(
                                x, hyp, null_hyp, rgba = False, order = 'fwd', plot=False)
            partial_woes = partial_woes.squeeze()
            print('Total WoE: {:4.2f} ({})'.format(total_woe.item(), hyp_woe.item()))

            ## Select which attributes to display
            idxs_pos = partial_woes.argsort()[-max_attrib:][::-1]
            idxs_pos = idxs_pos[partial_woes[idxs_pos] > 6]
            idxs_neg = partial_woes.argsort()[:max_attrib]
            idxs_neg = idxs_neg[partial_woes[idxs_neg] < -5]

            #max_S =  [a.tolist() for a in np.unravel_index(attrib_idxs[np.argmax(partial_woes)], (28,28))]
            max_S = np.unravel_index(attrib_idxs[np.argmax(partial_woes)], (28,28))
            max_S = np.s_[min(max_S[0]):max(max_S[0]),min(max_S[1]):max(max_S[1])]

            for typ,idxs in [('Positive', idxs_pos), ('Negative', idxs_neg)]:
                if len(idxs) > 0:
                    print(typ + ' Evidence:')
                    fig, axes = plt.subplots(1, len(idxs), figsize = (3*len(idxs),3))
                    for i,idx in enumerate(idxs):
                        ax = axes if len(idxs) ==1 else axes[i]
                        title = 'WOE: {:2.2f}'.format(partial_woes[idx])
                        plot_attribute(x.squeeze(), attrib_idxs[idx], ax = ax, title= title)
                    plt.show()

            ## Extract individual class woes for histogram plotting
            # Total woe hist
            class_total_woes = np.array([self.woe_model.woe(x, k, list(set(V) - set((k,)))).item() for k in V])
            # Attrib woe hist for top attrib
            class_attrib_woes = np.array([self.woe_model.woe(x, k, list(set(V) - set((k,))), subset=attrib_idxs[np.argmax(partial_woes)]).item() for k in V])

            hist = class_attrib_woes # class_attrib_woes

            idxs_srt = np.argsort(class_total_woes)[::-1]
            cum_hist_srt = [self.woe_model.woe(x, idxs_srt[:i].tolist(), idxs_srt[i:].tolist(),
            subset=attrib_idxs[np.argmax(partial_woes)]).item() for i in range(1,len(V))] + [0] # woe(V / emptyset ) is undefined
            cum_hist_srt = np.array(cum_hist_srt)
            #pdb.set_trace()

            #cum_hist = cum_hist_srt[j for i,j in enumerate(idx_srt)]
            cum_hist = 0*np.empty_like(hist)
            for i,j in enumerate(idxs_srt):
                cum_hist[i] = cum_hist_srt[j]
            #pdb.set_trace()

            #cum_hist = np.array([self.woe_model.woe(x, k, list(set(V) - set(), subset=attrib_idxs[np.argmax(partial_woes)]).item() for k in V])


            #hist = np.exp(hist - hist.max())
            #hist -= hist.min()
            #print(hist)
            #print(cum_hist)
            #pdb.set_trace()
            # k, hist = next(iter(self.woe_model.cache.items()))
            # assert k[1] == -1 # Want hist of full input
            # hist = np.array([hist[k].item() for k in sorted(hist) if k in V])
            # # Scale up by constant before exponentiate to avoid loss of precision
            # #print(hist)
            # #pdb.set_trace()
            # hist = np.exp(hist + np.abs(hist).mean())
            #pdb.set_trace()

            ## Remove classes from reference set
            V_prev = V.copy()
            V = [k for k in V if k in hyp]  # np version: np.isin(V, max_C)

            if verbose > 0:
                print('Shrinking ground set: {} -> {}'.format(V_prev, V))

            # Should make collection of results a dict to allow for different explainer
            # types returning different things. E.g., this has woe byt original doesn't
            predicate = AttrDict({'S': max_S, 'hyp': hyp, 'null_hyp': null_hyp,
                                  'hist': hist, 'cumhist': cum_hist, 'V_prev': V_prev, 'V': V, 'total_woe': total_woe})

            E.predicates.append(predicate)
            step += 1

        self.woe_model._stop_caching()
        return E

    def choose_hypothesis(self, x, y, V, reg_type = 'poly-centered', argmin_reg = None):
        """ Choose hypothesis (partition of reamining classes)

            Args:
                V: remaining classes to explain
                y: class predicted by the predictor

            Returns:
                The return value. True for success, False otherwise.
        """
        # add if catching torch setting:
        #self.woe_model.model.eval()
        #self.classifier.net.eval()
        #H, W = self.input_size
        #h, w = self.mask_size

        #pdb.set_trace(header='inside choose')
        if len(V) == 2:
            hyp = [y]
            null_hyp = [V[1]] if y == V[0] else [V[0]]
            return hyp, null_hyp, self.woe_model.woe(x, hyp, null_hyp)

        V = set(V)

        min_card = 1 #len(V)//2 - 1
        max_card = max(len(V)//2,2)
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            # TODO
            #return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(min_card, max_card+1))

        candidates = [set(C) for C in powerset(V) if y in C]

        scores = []
        # self.woe_model.caching = True
        # self.woe_model.cache = {hash(x) : {}}

        for C in candidates:
            woe = self.woe_model.woe(x, C, V.difference(C))
            # DEBUG
            argmin_reg = (len(V)+1)/3
            #
            if reg_type == 'decay':
                penalty = 1/np.power(len(C),p)
            elif reg_type == 'poly-centered':
                argmin = (len(V)+1)/2 if argmin_reg is None else argmin_reg
                #print(argmin)
                maxp   = max(np.abs((1 - argmin))**self.p, np.abs((len(C) - argmin))**self.p)
                penalty = (np.abs(len(C) - argmin)**self.p)/maxp  # Denom normalizes so that maxreg = 1
            else:
                raise ValueError('Unrecognized mode in normalized deltas')

            score =  woe - self.alpha*penalty
            scores.append((C, score, woe))

        # Sort in decreasing order - inplace
        scores.sort(key=lambda tup: tup[1], reverse = True)

        C_opt,score_opt, woe_opt = scores[0]
        print('Optimal score: {}'.format(score_opt.item()))
        #print('Optimal hypot: {}'.format(C_opt))

        return list(C_opt), list(V.difference(C_opt)), woe_opt

    def hypothesis_criterion(self, C, V, reg_type = 'decay', p = 2):
        """ TODO: Make this more efficient. I'm computing it for all cardinalities every time
        """
        k = len(V)
        Cards = np.arange(1,len(V)) # numpy and torch range have different behavior for top end of intervaal!
        if reg_type == 'decay':
            reg = 1/np.power(Cards,p)
        elif reg_type == 'poly-centered':
            argmin = (k+1)/2 if argmin_reg is None else argmin_reg
            #print(argmin)
            maxp   = max(np.abs((1 - argmin))**p, np.abs((len(Cards) - argmin))**p)
            reg = (np.abs(Cards - argmin)**p)/maxp  # Denom normalizes so that maxreg = 1
        else:
            raise ValueError('Unrecognized mode in normalized deltas')
        #print(reg)
        return reg[len(C)]

    #TODO: do we need static? Maybe make just normal with self, to avoid having to pass crit function
    @staticmethod
    def optimize_over_rectangle_attributes(model, criterion, masker, x, y, V = None, tgt_classes = [1,2],
                                 sort_hist = False, show_plot = False, plot_type = 'treemap', class_names = None,
                                 loss = 'euclidean', force_include_class = None, verbose = False):
        """
            Given masking model, input x and target histogram, finds subset of x (i.e., attribute)
            that minimizes loss with respect to target historgram.

             - V: ground set of possible classes
            plot: 0/1/2 (no plotting, plot only final, plot every iter)

        """
        model.net.eval()
        H, W = model.input_size
        h, w = model.mask_size
        max_obj, max_S = float("-inf"), None
        if V is None:
            V = np.array(list(range(10)))
        elif type(V) is list:
            V = np.array(V)
        if force_include_class:
            assert force_include_class in V, "Error: class to be force-included not in ground set"

        if KERNEL == 'terminal':
            #print("PLT ION")
            fig, ax = plt.subplots(1, 3, figsize = (4*3,4))
            plt.ion()
            plt.show()
            ims = None
        else:
            fig, ax, ims = None, None, None


        def handler(*args):
            print('Ctrl-c detected: will stop iterative plotting')
            nonlocal show_plot
            show_plot = min(show_plot, 1)

        #signal.signal(signal.SIGINT, handler)
        #print(KERNEL)
        if KERNEL == 'terminal': pbar = tqdm(total=(W-w)*(H-h))
        for (ii,jj) in itertools.product(range(0, W-w),range(0,H-h)):
                #print(ii,jj)
                if KERNEL == 'terminal': pbar.update(1)
                S = np.s_[ii:min(ii+h,H),jj:min(jj+w,W)]
                X_s, X_s_plot = masker(x, S)
                output = model(X_s)
                hist = output.detach().numpy().squeeze()
                hist_V = hist[V]
                obj, C_rel = criterion(hist_V, pred = y, V = V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)

                # Convert from relative indices to true classes
                 #TODO: Maybe move this to criterion?
                C = V[C_rel]

                labstr = 'Current objective: {:8.4e} (i = {:2}, j = {:2})'.format(obj.item(), ii, jj)
                if show_plot > 1:
                    if KERNEL == 'ipython':
                        clear_output(wait=True)
                        ax = None

                    ax, rects, ims = plot_2d_attrib_entailment(x, X_s_plot, S, C, V, hist_V,
                                        plot_type = plot_type, class_names = class_names,
                                        title = labstr, ax = ax, ims =ims, return_elements = True
                                        )

                    if KERNEL == 'terminal':
                        plt.draw()
                        plt.pause(.005)
                        #rects[0].remove()
                        ax[0].clear() # Alternatively, do rects[0].remove(), though its slower
                        ax[1].clear()
                        ax[2].clear()
                    else:
                        plt.show()
                elif verbose:
                    print(labstr)

                if force_include_class is not None and (not force_include_class in C):
                    # Target class not in entailed set
                    #print(force_include_class, C)
                    #print(obj, C_rel)
                    #global DEBUG
                    #DEBUG = True
                    #obj, C_rel = criterion(hist_V, pred = y, V = V)#, verbose = True)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
                    obj = float("-inf")
                    continue
                if obj > max_obj:
                    max_obj = obj
                    max_S   = S

        if KERNEL == 'terminal': pbar.close()
        if max_S is None:
            print('Warning: could not find any attribute that causes the predicted class be entailed')
            return None, None, None

        if KERNEL == 'terminal':
            plt.ioff()
            plt.close('all')

        # Reconstruct vars for optimal
        X_s, X_s_plot = masker(x, max_S)
        output = model(X_s)
        hist = output.detach().numpy().squeeze()
        hist_V = hist[V]
        obj, C_rel = criterion(hist_V, pred = y, V=V)#, sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
        max_C = V[C_rel]

        ii, jj =  list(range(H)[max_S[0]])[0], list(range(W)[max_S[1]])[0]

        assert obj == max_obj
        labstr = 'Best objective: {:8.4f} (i = {:2}, j = {:2})'.format(max_obj.item(), ii, jj)
        #clear_output(wait=True)
        if show_plot:
            plot_2d_attrib_entailment(x, X_s_plot, max_S, max_C, V, hist_V, plot_type=plot_type,
            class_names = class_names,
            topk = 10, title = labstr, sort_bars = True)
            plt.show()
        else:
            print(labstr)

        return max_S, max_C, hist_V

    @staticmethod
    def optimize_over_ngram_attributes(model, criterion, masker, x, y, V = None,
                                 tgt_classes = [1,2], sort_hist = False,
                                 show_plot = False, plot_type = 'treemap',
                                 class_names = None, loss = 'euclidean',
                                 force_include_class = None, verbose = False):
        """
            Given masking model, input x and target histogram, finds subset of x
            (i.e., attribute) that minimizes loss with respect to target historgram.

             - V: ground set of possible classes
            plot: 0/1/2 (no plotting, plot only final, plot every iter)

        """
        model.net.eval()
        N = x.shape[1]
        n = model.mask_size
        max_obj, max_S = 100*torch.ones(1), None
        if V is None:
            V = np.array(list(range(10)))
        elif type(V) is list:
            V = np.array(V)
        if force_include_class:
            assert force_include_class in V, "Error: class to be force-included not in ground set"

        if KERNEL == 'terminal':
            fig, ax = plt.subplots(1, 2, figsize = (10,4),
                                gridspec_kw = {'width_ratios':[2, 1]})
            plt.ion()
            plt.show()
            ims = None
        else:
            fig, ax, ims = None, None, None

        def handler(*args):
            print('Ctrl-c detected: will stop iterative plotting')
            nonlocal show_plot
            show_plot = min(show_plot, 1)

        signal.signal(signal.SIGINT, handler)

        for ii in range(0, N-n):
            #print(ii)
            S = np.s_[range(ii,ii+n)]
            X_s    = masker(x, S)
            output = model(X_s)
            hist = output.detach().numpy().squeeze()
            hist_V = hist[V]
            obj, C_rel = criterion(hist_V, pred = y, V = V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
            #obj, C_rel = criterion(hist_V, V=V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
            # Convert from relative indices to true classes
             #TODO: Maybe move this to criterion?
            C = V[C_rel]
            labstr = 'Ngram: [{}:{}], Objective: {:8.4e}'.format(ii, ii+n,obj.item())
            ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
            if show_plot > 1:
                if KERNEL == 'ipython':
                    clear_output(wait=True)
                    ax = None

                ax = plot_text_attrib_entailment(x, X_s, S, C, V, hist_V,
                                    plot_type = plot_type,
                                    class_names = class_names,
                                    vocab = model.vocab, ax = ax,
                                    title = labstr, sort_bars = False)
                if KERNEL == 'terminal':
                    plt.draw()
                    plt.pause(.005)
                    ax[0].clear()
                    ax[1].clear()
                else:
                    plt.show()
            elif verbose:
                #ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
                #labstr = 'Ngram: {} ([{}:{}]), Objective: {:8.4e}'.format(ngram_str, ii, ii+n,obj.item())
                print(labstr)

            if force_include_class is not None and (not force_include_class in C):
                # Target class not in entailed set
                #print(force_include_class, C)
                obj = float("-inf")
                continue
            if obj < max_obj:
                max_obj = obj
                max_S   = S
                #clear_output(wait=True)

        if max_S is None:
            print('Warning: could not find any attribute that causes to predicted class be entailed')
            return None, None, None

        if KERNEL == 'terminal':
            plt.ioff()
            plt.close('all')

        # Reconstruct vars for optimal
        X_s = masker(x, max_S)
        output = model(X_s)
        hist = output.detach().numpy().squeeze()
        hist_V = hist[V]
        obj, C_rel = criterion(hist_V, pred = y, V = V)#, sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
        max_C = V[C_rel]
        #print(asd.asd)
        ii = max_S[0]

        assert obj == max_obj
        ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
        labstr = 'Best objective: {:8.4f}, (i = {:2})'.format(max_obj.item(),ii)
        #clear_output(wait=True)
        if show_plot:
                plot_text_attrib_entailment(x, X_s, max_S, max_C, V, hist_V,
                                    plot_type = plot_type,
                                    class_names = class_names,
                                    vocab = model.vocab,
                                    title = labstr, sort_bars = False)
                plt.tight_layout()
                plt.show()
        else:
            print(labstr)
        return max_S, max_C, hist_V
