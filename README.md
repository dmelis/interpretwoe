# mce
Multistep Contrsative Explanations

## Dependencies

#### Major:
* pytorch (>4.0)
* [faiss](https://github.com/facebookresearch/faiss) (for fast nearest neighbor search, optional -needed only if retraining mask model)

#### Minor:
* numpy
* matplotlib
* nltk (needed only for text applications)
* shapely
* squarify

Installing Pytorch. Find approriate version download link [here](https://pytorch.org/) e.g.:

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

## Setting up

After cloning, do:
```
  python setup.py install
```

Download Pretrained Models:

```
wget  people.csail.mit.edu/davidam/MCE/checkpoints/mnist/classif.pth -P checkpoints/mnist/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/mnist/mask_model_7x7.pth -P checkpoints/mnist/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/classif.pth -P checkpoints/hasy/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/mask_model_10x10.pth -P checkpoints/hasy/

```

Run Demo:

```
  python main_mnist.py
```
