# autotres

[![Build Status](https://travis-ci.org/arizona-phonological-imaging-lab/autotres.svg?branch=master)](https://travis-ci.org/arizona-phonological-imaging-lab/autotres)

## What is it?
`autotres` is a collection of tools to analyze tongue surface contours in ultrasound images.



## Where did it come from?
`autotres` is a direct descendant of `AutoTrace` (AutoTrace III).  Read about the original in Jeff Berry's dissertation:

```
@phdthesis{berry2012diss,
  title={Machine learning methods for articulatory data},
  author={Berry, Jeffrey James},
  year={2012},
  school={The University of Arizona.}
}
```

# Installation

## Dependencies

We recommend running the project using a virtual environment. `virtualenv` can be installed with `pip install virtualenv`.

1. `virtualenv -p python3 venv`
2. `source venv/bin/activate`

## Installation instructions:

### System dependencies


#### OSX

The system dependencies can be installed via `homebrew`:

```
brew update;
brew install python3;
brew install gfortran;
brew tap homebrew/science;
brew install openblas;
brew install hdf5;
```

#### Linux

```
sudo apt-get update;
sudo apt-get install build-essential;
sudo apt-get install gcc;
sudo apt-get install python3-dev;
sudo apt-get install libhdf5-dev;
sudo apt-get install gfortran;
sudo apt-get build-dep libopenblas-dev;
sudo apt-get build-dep nvidia-cuda-toolkit;
```

### Installing `autotres` and its Python dependencies:

```
pip install -e .
```

# Using `autotres`

We provide network training and usage tutorials under [examples](examples).


## Example data

You will need [`git-lfs`](https://git-lfs.github.com) to pull the example data.

Once you've installed `git-lfs`, simply run this command:

```
git-lfs fetch
```

## Training networks

Networks can be trained using either a GPU or CPU.  GPU training will save a great deal of time.

### GPU-based training

The code for training deep networks uses [`Lasagne`](https://github.com/Lasagne/Lasagne), a wrapper for [`Theano`](http://deeplearning.net/software/theano/).  One of the advantages of relying on these libraries is that networks can easily be trained on a CUDA-capable GPU, if present (with limited support for `open-cl`).

### CPU-based training

If no GPU is present, `Theano` will use the CPU. For best results, a `BLAS` library with multithreading support is suggested, such as ['BLAS'](http://www.netlib.org/blas/) or [`OpenBLAS`](http://www.openblas.net).


# What's missing?

Currently, the project lacks a graphical interface, and has only been tested on Ubuntu 14.04 and OSX 10.11. With luck, future versions will rectify these shortcomings.
