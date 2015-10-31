#!/usr/bin/env python3

import os
import logging
from glob import glob
import fnmatch
import h5py
from PIL import Image
import numpy as np

from .roi import ROI

def get_from_files(d,path,roi,scale=1,n_points=32,buff=512,blacklist=[]):
    """Create an hdf5 dataset from a folder of images and traces

    Tries to match names of traces with names of images.  
    Args:
        d (str): The path of a folder.
            The folder is recursively searched.
        path (str): Where to save the dataset
            Any existing file will be overwritten without warning
        roi (ROI): The partof each image to extract.
        scale (numeric, optional):
            A factor by which to scale the images.
            Defaults to 1 (no scaling). A better setting might be 0.1
        n_points (int, optional): The number of points in each trace
            Defaults to 32
        buff (int, optional): Number of images to buffer before writing
            Defaults to 512
        blacklist (container): Set of image filenames to ignore
            This is particularly useful for making disjoint training / 
                testing datasets
            Defaults to the empty list (i.e. nothing excluded)
    """
    images = []
    traces = []
    names = []
    roi = ROI(roi)
    roi_s = roi.scale(scale)
    if os.path.exists(path):
        os.remove(path)
    hp = h5py.File(path,'w')
    hp.create_dataset('image',
        (0,1) + roi_s.shape,
        maxshape = (None,1) + roi_s.shape,
        chunks = (buff,1) + roi_s.shape, compression='gzip')
    hp.create_dataset('trace',
        (0,n_points,1,1),
        maxshape = (None,n_points,1,1),
        chunks = (buff,n_points,1,1), compression='gzip')
    try:
        unicode
    except NameError:
        unicode = str
    hp.create_dataset('name',
        (0,),
        maxshape = (None,),
        chunks = (buff,),
        dtype=h5py.special_dtype(vlen=unicode), compression='gzip')
    # traverse d 
    for root,__,filenames in os.walk(d):
        # look for hand-traced traces
        for filename in fnmatch.filter(filenames,'*.ghp.traced.txt'):
            # because it matched the above fnmatch, we can assume it 
            # ends with '.ghp.traced.txt' and remove that ending.
            # the rest is our target
            base = filename[:-len('.ghp.traced.txt')]
            # look for our target
            f = None
            if os.path.isfile(os.path.join(root,base)):
                f = os.path.join(root,base)
            else:
                g = glob(os.path.join(root,'..','[sS]ubject*','IMAGES',base))
                if g:
                    f = g[0]
            # if we found it, then put it and our trace in the list
            if f:
                if os.path.basename(f) not in blacklist:
                    image = image_from_file(f,roi,scale)
                    trace = trace_from_file(os.path.join(root,filename),
                        roi,n_points)
                    try:
                        if image.any() and trace.any():
                            images.append(image)
                            traces.append(trace)
                            names.append( os.path.basename(f) )
                    except AttributeError:
                        logging.error("%s %s" % (image, trace))
                        raise
                else:
                    logging.debug("excluding file %s" % (os.path.basename(f)))
            if len(images) >= buff:
                s = hp['image'].shape[0]
                images_add = np.array(images[:buff],dtype='float32')
                traces_add = np.array(traces[:buff],dtype='float32')
                hp['image'].resize(s+buff,0)
                hp['image'][s:] = images_add
                hp['trace'].resize(s+buff,0)
                hp['trace'][s:] = traces_add 
                hp['name'].resize(s+buff,0)
                hp['name'][s:] = names[:buff] 
                images = images[buff:]
                traces = traces[buff:]
                names = names[buff:]
                logging.info( "image: %s trace: %s name %s" %
                    (hp['image'].shape, hp['trace'].shape, hp['name'].shape))
    logging.info( "image: %s trace: %s name %s" %
        (hp['image'].shape, hp['trace'].shape, hp['name'].shape))
    hp.close()

                
def image_from_file(f,roi,scale=.01):
    """Extract a porperly scaled section of an image

    Args:
        f (str): The path to an image
        roi (ROI): The part of the image to extract
        scale
    """
    roi = ROI(roi)
    roi_scale = roi.scale(scale)
    img = Image.open(f)
    img = img.convert('L')
    img.thumbnail((img.size[0] * scale, img.size[1] * scale))
    img = np.array(img,dtype='float32')
    img = img / 255
    img = np.array(img[roi_scale.slice],dtype='float32')
    img = img.reshape(1,img.shape[0],img.shape[1])
    return img


def trace_from_file(fname,roi,n_points):
    """Extract a trace from a trace file

    Uses a linear interpolation of the trace to extract evenly-spaced points
    Args:
        fname (str): The path to a trace file.
        roi (ROI): The space accross which to evenly space the points
        n_points (int): The nuber of points to extract
    """
    roi = ROI(roi)
    gold_xs = []
    gold_ys = []
    with open(fname) as f:
        for l in f:
            l = l.split()
            if int(l[0]) > 0:
                gold_xs.append(float(l[1]))
                gold_ys.append(float(l[2]))
    gold_xs = np.array(gold_xs,dtype='float32')
    gold_ys = np.array(gold_ys,dtype='float32')
    if len(gold_xs) > 0: 
        trace = np.interp(roi.domain(n_points),gold_xs,gold_ys,left=0,right=0)
        trace = trace.reshape((n_points,1,1))
        trace[trace==0] = roi.offset[0]
        trace = (trace - roi.offset[0]) / (roi.height)
    else:
        return np.array(0)
    if trace.sum() > 0 :
        return trace
    else: 
        return np.array(0)

