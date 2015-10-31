# autotres

## What is it?
A collection of tools to analyze tongue surface contours in ultrasound images.

Autotres is a direct descendant of AutoTrace (AutoTrace III).  Read about the original in Jeff Berry's dissertation:

```
@phdthesis{berry2012diss,
  title={Machine learning methods for articulatory data},
  author={Berry, Jeffrey James},
  year={2012},
  school={The University of Arizona.}
}
```

The code for training deep networks uses [`Lasagne`](https://github.com/Lasagne/Lasagne) and [Theano](http://deeplearning.net/software/theano/)
These allow the network to be trained on a CUDA-capable GPU if present (with limited support for open-cl).
If no GPU is present, `theano` will use the CPU. For best results, a `BLAS` library with multithreading support is suggested, such as [`OpenBLAS`](http://www.openblas.net).

Currently, the project lacks a graphical interface, and has only been tested on Ubuntu 14.04. With luck, future versions will rectify these shortcomings.
