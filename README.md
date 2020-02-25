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
* attrdict

## Installation

It's highly recommended that the following steps be done **inside a virtual environment** (e.g., via `virtualenv` or `anaconda`).


#### Install prereqs

Installing Pytorch. Find approriate version download link [here](https://pytorch.org/) e.g.:

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl  # For CPU
# pip3 install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl # For GPU - CUDA 9.0 in python 3.6
# pip3 install https://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl # For GPU - CUDA 9.1 in python 3.6
# etc.....
pip3 install torchvision
```
Then install remaining dependencies
```
pip3 install -r requirements.txt
```
Finally, install this package
```
git clone git@github.com:dmelis/mce.git
cd mce
pip3 install -e ./
```

## Data preparation:

Invoke the makefile with the desired dataset as argument (options currently supported: [`ets`, `hasy`,`leafsnap`]), e.g.:

```
make hasy

```

Or generate all of them with `make all`.

NOTE: Since the ETS data is from LDC (and thus not public), I hid it under a password in my website. Ask me and I'll provide it directly. After executing `make ets` you'll be prompted for this password.

<!-- ```
  python setup.py install
``` -->

## How to use

To train models from scratch:
```
python scripts.main_mnist.py --train_classif --train_meta
```
<!-- ```
python -m scripts.main_mnist --train-classif --train-meta
``` -->

To use pretrained models:
```
python scripts/main_mnist.py
```

The first time main_mnist.py is run, it will download MNIST (takes a few minutes).


The first time main_ets.py is run, it will download golve embeddings (takes a few minutes).


## Overall Code Structure


* explainers.py - defines the Explanation, Explainer Classes, contains all "explanation scoring functions"
* models.py - collection of pytorch nn Modules defining classifier and masked classifier architectures
* datasets.py - utils for loading datasets
* methods.py -


## Classes and methods

* woe_wrapper() -> parent class for woe wrappers, torch/numpy agnostic
  - woe_scikit_gnb() -> for Gaussan naive bayes model

<!-- Otherwise, download Pretrained Models:

```
wget  people.csail.mit.edu/davidam/MCE/checkpoints/mnist/classif.pth -P checkpoints/mnist/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/mnist/mask_model_7x7.pth -P checkpoints/mnist/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/classif.pth -P checkpoints/hasy/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/mask_model_10x10.pth -P checkpoints/hasy/ -->
