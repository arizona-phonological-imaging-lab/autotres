# autotres

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

All of the python dependencies are listed under [`requirements.txt`](requirements.txt) and can be installed with the following command:

```
pip install -r requirements.txt
```
# Using `autotres`

We provide network training and usage tutorials under [examples](exmaples).

## Training networks

Networks can be trained using either a GPU or CPU.  GPU training will save a great deal of time.

### GPU-based training

The code for training deep networks uses [`Lasagne`](https://github.com/Lasagne/Lasagne), a wrapper for [`Theano`](http://deeplearning.net/software/theano/).  One of the advantages of relying on these libraries is that networks can easily be trained on a CUDA-capable GPU, if present (with limited support for `open-cl`).

### CPU-based training

If no GPU is present, `Theano` will use the CPU. For best results, a `BLAS` library with multithreading support is suggested, such as [`OpenBLAS`](http://www.openblas.net).


# What's missing?

Currently, the project lacks a graphical interface, and has only been tested on Ubuntu 14.04. With luck, future versions will rectify these shortcomings.
