# Weight-of-Evidence Interpretability 

Codebase accompanying the papers:
* [Weight of Evidence as a Basis for Human-Oriented Explanations](https://arxiv.org/abs/1910.13503).
* [From Human Explanation to Model Interpretability: A Framework Based on Weight of Evidence](https://arxiv.org/abs/2104.13299).

See the papers for technical details. 

## Dependencies

* python (>3.0)
* numpy
* scipy
* pandas 
* scikit-learn
* matplotlib
* seaborn
* attrdict
* colored
* tqdm
* jupyterlab (optional - to run notebooks )
* pytorch (optional - to use with pytorch models: BETA)

## Installation

It's highly recommended that the following steps be done **inside a virtual environment** (e.g., via `virtualenv` or `anaconda`).


#### Via Conda (recommended)

If you use [ana|mini]conda , you can simply do:

```
conda env create -f environment.yaml python=3.8
conda activate interpretwoe
conda install . # (optional: to import without needing to add path)
```

#### Via pip

```
pip install -r requirements.txt
```
Finally, install this package:
```
pip install .
```


## How to use

For example usage, please see `notebooks/WoE_UserStudy_Main.ipynb`. That notebook has a self-contained full experimental setup, which we used for our user study.


## Overall Code Structure

The main relevant code is in the following scripts in `src/':

* explainers.py - defines the Explanation, Explainer Classes, etc
* scoring.py - defines the explanation scoring function for choose class partitions
* woe.py - defines weight-of-evidence computation methods
* data.py - data loading functions
* classifiers.py - classification models
* woe_utils.py - misc utils used by woe models


