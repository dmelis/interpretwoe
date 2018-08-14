from __future__ import division
import warnings
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pdb

def not_inline_warning():
    backend = matplotlib.get_backend()
    if "backend_inline" not in backend:
        warnings.warn("livelossplot requires inline plots.\nYour current backend is: {}\nRun in a Jupyter environment and execute '%matplotlib inline'.".format(backend))

# TODO
# * object-oriented API
# * only integer ticks

def draw_plot(logs, plt_logs, metrics, plottypes, figsize=None, max_epoch=None,
              max_cols=2,
              validation_fmt="val_{}",
              metric2title={}):
    clear_output(wait=True)
    plt.figure(figsize=figsize)

    panes = len(metrics) + len(plottypes) + 1

    for metric_id, metric in enumerate(metrics):
        plt.subplot((panes) // max_cols + 1, max_cols, metric_id + 1)

        if max_epoch is not None:
            plt.xlim(1, max_epoch)

        plt.plot(range(1, len(logs) + 1),
                 [log[metric] for log in logs],
                 label="training")

        if validation_fmt.format(metric) in logs[0]:
            plt.plot(range(1, len(logs) + 1),
                     [log[validation_fmt.format(metric)] for log in logs],
                     label="validation")

        plt.title(metric2title.get(metric, metric))
        plt.xlabel('epoch')
        #plt.legend(loc='center right')

    for plot_id, plottype in enumerate(plottypes):
        plt.subplot((panes) // max_cols + 1, max_cols, plot_id + metric_id + 2)
        #pdb.set_trace()
        if 'imshow' in plottype:
            plt.imshow(plt_logs[-1][plottype])
            plt.axis('off')
        elif 'bar' in plottype:
            plt.bar(range(10),plt_logs[-1][plottype])
            plt.ylim(0,1)
            plt.xticks(range(10), range(10))

    #print(asd.asdas)
    plt.tight_layout()
    plt.show();

class PlotLosses():
    def __init__(self, figsize=None, cell_size=(6, 4), dynamic_x_axis=False, max_cols=2, max_epoch=None, metric2title={},
    validation_fmt="val_{}", imshow_fmt = "imshow_{}"):
        self.figsize = figsize
        self.cell_size = cell_size
        self.dynamic_x_axis = dynamic_x_axis
        self.max_cols = max_cols
        self.max_epoch = max_epoch
        self.metric2title = metric2title
        self.validation_fmt = validation_fmt
        self.imshow_fmt = imshow_fmt
        self.logs = None

        not_inline_warning()

    def set_metrics(self, metrics, plottypes):
        self.base_metrics = metrics
        self.plottypes    = plottypes
        if self.figsize is None:
            self.figsize = (
                self.max_cols * self.cell_size[0],
                ((len(self.base_metrics) + len(self.plottypes) + 1) // self.max_cols + 1) * self.cell_size[1]
            )

        self.logs = []
        self.plt_logs = []
        #pdb.set_trace()

    def update(self, log):
        loss_log = {k:v for k,v in log.items() if not 'plt' in k}
        plt_log = {k:v for k,v in log.items() if 'plt' in k}

        if self.logs is None:
            metrics = [metric for metric in log.keys() \
             if not 'val' in metric.lower() and \
             not 'plt' in metric.lower()]
            plottypes =  [metric for metric in log.keys() \
             if  'plt' in metric.lower()]
            self.set_metrics(metrics, plottypes)
        # if self.plt_logs is None:
        #     self.set_metrics([metric for metric in log.keys() \
        #      if not 'val' in metric.lower() and \
        #      not 'imshow' in metric.lower()])
        #print(plt_log.keys())
        self.logs.append(loss_log)
        self.plt_logs.append(plt_log)



    def draw(self):
        draw_plot(self.logs, self.plt_logs, self.base_metrics, self.plottypes,
                  figsize=self.figsize, max_epoch=self.max_epoch,
                  max_cols=self.max_cols,
                  validation_fmt=self.validation_fmt,
                  metric2title=self.metric2title)
