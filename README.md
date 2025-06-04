# Intro to ML Basics:

## Types of ML: Supervised, Unsupervised, Reinforcement

There are three main types of machine learning:

- **Supervised Learning:** uses labeled data and direct feedback to predict future outcomes

  - Classification (Binary and Multi-class): Predicting food type (pizza, steak, pasta) from features
  - Regression: (Predicting house-price based on home features)

- **Unsupervised Learning:** data doesn't use labels or targets to find patterns in the data set

  - Clustering and Dimensionality Reduction

- **Reinforcement Learning:** model improves its performance based on interactions with the environment and a reward system
  - Think a game of chess where the moves are recorded and the ultimate win/loss outcome is the reward

## Common Notation / Terminology:

- Training example: a row in a table representing the dataset and synonymous with an observation, record, instance
  or sample (a collection of training examples)

- Training: model fitting

- Feature (x): a column in a table representing the dataset and synonymous with predictor, variable, input,
  attribute or covariate

- Target (y): synonymous with outcome, output, response variable, dependent variable, (class) label

- Loss Function: AKA `cost` function. Somtimes referred to as an `error function` and is a measurement of the
  distance from the target/objective measure

## General ML Steps:

1. Data Collection
2. Data Pre-processing (missing data, initial feature extraction, selection, scaling, one-hot encoding,
   dimensionality reduction, data augmentation, etc.)
3. Data Splitting (Training/Validation/Test)
4. Model Building (Layers, loss function, optimization variables)
5. Model Training
6. Model Evaluation (Inference Predictions)
7. Iterate on Model Features (Step 4. - i.e fine-tuning layers or adding/removing layers)

> _No Free Lunch Theorem_:
> "...if the only tool you have is a hammer its tempting to treat everything as if it were a nail"
> Each **CLASSIFICATION** algorithm has inherent biases
> No single classification model enjoys superiority if we don't make assumptions about the task.
> In practice--it is essential to compare a handful of different learning algorithms in order to train and select the best performing model.
> FIRST: choose a performance metric (often **classification accuracy**)

# ML Environment Setup

## Python

### 1. Create a python virtual environment

```bash
python -m .venv venv
```

### 2. Activate the virtual environment

```bash
source .ven/bin/activate
```

### 3. Install a python package

```bash
pip install <package>
```

### 4. Upgrade python package

```bash
pip install <package> --upgrade
```

## Anaconda Python Distro and Package Manager

**Anaconda Installer**

https://docs.anaconda.com/anaconda/install

**Miniconda Installer (RECOMMENDED)**

- A leaner alternative to Anaconda that comes without any packages pre-installed

https://docs.conda.io/en/latest/miniconda.html

### 1. Install Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### 2. Manually add `conda` to `PATH`

Allows you access to the `conda` command in the CLI without defaulting your terminal bash sessions to a conda (base) env

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

### 3. Create a conda virtual environment

```bash
conda create -n <env_name> python=<version>
```

example: conda create -n chapter1 python=3.9

### 4. Activate conda virtual env

```bash
conda activate <env_name>
```

### 5. Install package in Miniconda Virtual Env

```bash
conda install <package>
```

### 6. Update Existing Packages

```bash
conda update <package>
```

### **Package Availability**

Packages not available through the official conda channel may be available via the community supported conda-forge project (https://conda-forge.org)

#### 1. Conda-Forge Package Installation

```bash
conda install <package> --channel conda-forge

```

Packages not available through the offical conda channel or conda-forge can be installed via pip

#### 2. Pip Installation

```bash
pip install <package>
```

# Data Science & ML Packages

## NumPy

- we will use NumPy's multi-dimensional arrays to store and manipulate data

## Pandas

- we will use Pandas (dataframes) to help with manipulating tabular data

## Matplotlib

- We will use Matplotlib to visualize quantitative data

## Scikit-learn

- Main machine learning library

## PyTorch

- Used for building Deep-Learning Models

> Library Version Control
> Note: the library versions used in the book are listed below and should be the installed versions to ensure
> compatibility
>
> - NumPy 1.21.2
> - SciPy 1.70
> - Scikit-learn 1.0
> - Matplotlib 3.4.3
> - pandas 1.3.2

Note: after installing the packages you can double check the version number by accessing the **version** attribute:

```python
import numpy
print(numpy.__version__)
```

Note: the authors included a `python-environment-check.py` script for checking both the Python version and the
package versions at: https://github.com/rasbt/machine-learning-book
