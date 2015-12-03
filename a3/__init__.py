#!/usr/bin/env python3

from __future__ import absolute_import

# These evironment flags set up GPU training, 
# but we don't want to polute our namespace, so private var
import os as __os
__os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu'

from .roi import ROI
from .autotracer import Autotracer
from .lib import get_from_files, image_from_file, trace_from_file
from .errors import ShapeError, ConflictError
from .constants import *
from .dataset import Dataset
