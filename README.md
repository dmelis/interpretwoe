# mce
Multistep Contrsative Explanations

## Dependencies

#### Major:
* python (>3.0)
* pytorch (>4.0)
* (OPTIONAL) [faiss](https://github.com/facebookresearch/faiss) (for fast nearest neighbor search, needed only if retraining mask model)

#### Minor
* numpy
* matplotlib
* nltk (needed only for text applications)
* torchtext (needed only for text applications)
* shapely
* squarify

Installing Pytorch. Find approriate version download link [here](https://pytorch.org/) e.g.:

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

## Install
After cloning, do:
```
pip3 install -r requirements.txt
```

## Data preparation:

Invoke the makefile with the desired dataset as argument (options currently supported: [`mnist`, `hasy`,`leafsnap`]), e.g.:

```
make hasy

```

Or generate all of them with `make all`

<!-- ```
  python setup.py install
``` -->

## How to use

To train models from scratch:
```
python -m scripts.main_mnist --train-classif --train-meta
```

To use pretrained models:
```
python -m scripts.main_mnist
```
<!-- Otherwise, download Pretrained Models:

```
wget  people.csail.mit.edu/davidam/MCE/checkpoints/mnist/classif.pth -P checkpoints/mnist/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/mnist/mask_model_7x7.pth -P checkpoints/mnist/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/classif.pth -P checkpoints/hasy/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/mask_model_10x10.pth -P checkpoints/hasy/ -->
