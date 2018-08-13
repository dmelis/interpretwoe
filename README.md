# mce
Multistep Contrsative Explanations

Preeqs:

Major:
* pytorch (>4.0)
* faiss (for fast nearest neighbor search, optional -needed only if retraining mask model)

Minor:
* numpy
* matplotlib
* nltk (needed only for text applications)
* shapely
* squarify

Installing Pytorch, e.g.:

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```



Download Pretrained Models

```
wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/classif.pth -P src/hasy/

wget  people.csail.mit.edu/davidam/MCE/checkpoints/hasy/mask_model_10x10.pth -P src/hasy/

```
