#!/usr/bin/env python3

class ShapeError(Exception):
    """Raised when invalid in/output dimensionality causes an error

    Typically, this means that the network tried to compile before
    input or output dimensionality has been set, or with invalid shape.
    Can also mean that input or output shape has been set, but doesn't
    match the shape of the data.
    Attributes:
        Xshape (tuple of int): the shape of the input data (X)
            A value of None means that input shape has not been set.
        yshape (tuple of int): the shape of the output data (y)
            A value of None means that output shape has not been set.
    """
    def __init__(self,Xshape,yshape):
        self.Xshape = Xshape
        self.yshape = yshape

class ConflictError(Exception):
    """Raised when there is conflicting data for the same token

    This means that you have two conflicting traces, images, etc.
    for a certain token. This could also mean you have non-unique IDs.
    Attributes:
        ID (str): the identifier for which conflicting data exists
        f1 (str): the path to the first file
        f2 (str): the path to the file that conflicts with f1
    """

    def __init__(self,ID,f1,f2):
        self.ID = ID
        self.f1 = f1
        self.f2 = f2
