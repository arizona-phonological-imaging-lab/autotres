#!/usr/bin/env python3

from __future__ import division

import numpy as np
import json

class ROI(object):
    """ Region of Interest for a set of images
    
    Attributes:
        shape (tuple of numeric): the height and width of the ROI
        offset (tuple of numeric): the lower bounds of the ROI
        extent (tuple of numeric): the upper bounds of the ROI
            offset[dim] + shape[dim] should always == extent[dim]
        orthodox (tuple of bool): whether the ROI is indexed "normally"
            I.e. if the ROI is measured from the top/left
            If measured from the bottom-left: (False, True)
        slice (tuple of slice): can be used to slice into a 2d matrix
            >>> np.identity(5)[ROI(2,3,1,4).slice]
            array([[ 0., 1., 0.]])
        
    """
    def __init__(self,*args,**kwargs):
        """
        Multiple possible ways of declaring an ROI are supported.
        The first way is by specifying the bounds as positional args
        Args:
            top (numeric): the top of the region of interest
            bottom (numeric): the bottom of the region of interest
            left (numeric): the left edge of the region of interest
            right (numeric): the right edge of the region of interest
        Example:
            >>> ROI(1,2,3,4)
            ROI(1.0, 2.0, 3.0, 4.0)

        The second way is by specifying a single iterable object
        Example:
            >>> ROI(1,2,3,4) == ROI([1,2,3,4])
            True

        Regardless of the constructor format used, the order should
            always be: top, bottom, left, right
        This allows for symantic interpretation of the arguments.
            ROI is smart enough to deal with indexing from other edges
        Example:
            >>> ROI(2,1,4,3).slice
            (slice(1.0, 2.0, None), slice(3.0, 4.0, None))
            >>> ROI(2,1,4,3).top
            2.0
        """
        if len(args) == 4:
            roi = (args[0],args[1],args[2],args[3])
        elif len(args) == 1:
            roi = args [0]
        (top, bottom, left, right) = [float(x) for x in roi]
        self.orthodox = (top<bottom, left<right)
        self.shape  = (abs(top-bottom), abs(left-right))
        self.offset = (min(top,bottom), min(left,right))
        self.extent = (max(top,bottom), max(left,right))
        self.slice = (slice(self.offset[0],self.extent[0]),
            slice(self.offset[1],self.extent[1]))

    @property
    def top(self): 
        """Convenience property for the top of the ROI
            For an orthodox ROI, this is the same as offset[0]
            For an ROI unorthodox in the Y dimension, this is extent[0]
        """
        return self.offset[0] if self.orthodox[0] else self.extent[0]
    @property
    def bottom(self): 
        """Convenience property for the bottom of the ROI
            For an orthodox ROI, this is the same as extent[0]
            For an ROI unorthodox in the Y dimension, this is offset[0]
        """
        return self.extent[0] if self.orthodox[0] else self.offset[0]
    @property
    def left(self): 
        """Convenience property for the left of the ROI
            For an orthodox ROI, this is the same as offset[1]
            For an ROI unorthodox in the X dimension, this is extent[1]
        """
        return self.offset[1] if self.orthodox[1] else self.extent[1]
    @property
    def right(self): 
        """Convenience property for the right of the ROI
            For an orthodox ROI, this is the same as extent[1]
            For an ROI unorthodox in the X dimension, this is offset[1]
        """
        return self.extent[1] if self.orthodox[1] else self.offset[1]
    @property
    def height(self): 
        """Convenience property for the height of the ROI 
            This is the same as shape[0]
        """
        return self.shape[0]
    @property
    def width(self): 
        """Convenience property for the width of the ROI
            This is the same as shape[1]
        """
        return self.shape[1]

    def __repr__(self):
        return 'ROI(%s, %s, %s, %s)' % tuple(self)
    
    def __eq__(self,other):
        return repr(self) == repr(other)

    def __iter__(self):
        """Iterate over ROI bounds

        Yields:
            numeric: top, bottom, left, right (strictly ordered)
        """
        return (x for x in (self.top,self.bottom,self.left,self.right))

    def domain(self,N):
        """Returns a numpy array of N equally-spaced x values in the ROI
        
        Args:
            N (integer): number of points to create

        Returns:
            numpy array: N evenly-spaced points, from offset[1] to 
                extent[1] (includes neither offset[1] nor extent[1])
                The dtype should be float32

        Example:
            >>> ROI(x,y,10,20).domain(3)
            array([12.5,15.,17.5])
        """
        step = self.shape[1] / (N + 1)
        return np.arange(self.offset[1] + step, self.extent[1], step)
    
    def json(self):
        """json stringify the ROI"""
        j = {
            'srcY': self.offset[0],
            'destY': self.shape[0],
            'srcX': self.offset[1],
            'destX': self.shape[1],
        }
        return json.dumps(j)
    

    def scale(self,factor):
        """Create a scaled version of the current ROI.
        
        Args:
            factor (numeric): the factor by which to scale. 
        
        Returns:
            ROI: the scaled ROI

        Example:
            >>> ROI(1,2,3,4).scale(2.5)
            ROI(2.5, 5.0, 7.5, 10.0)
        """
        return ROI(np.array(tuple(self))*factor)
