#!/usr/bin/env python3

import os
import logging
from glob import glob
import fnmatch
import h5py
from PIL import Image
import numpy as np
import pyglet

from .roi import ROI

def image_from_file(path,roi,scale=1,**kwargs):
    """Extract a porperly scaled section of an image

    Args:
        path (str): The path to an image
        roi (ROI): The part of the image to extract
        scale (float in (0-1]): Scaling factor
    """
    roi = ROI(roi)
    roi_scale = roi.scale(scale)
    f = Image.open(path)
    img = f.convert('L')
    f.close
    img.thumbnail((img.size[0] * scale, img.size[1] * scale))
    img = np.array(img,dtype='float32')
    img = img / 255
    img = np.array(img[roi_scale.slice],dtype='float32')
    img = img.reshape(1,img.shape[0],img.shape[1])
    return img

def trace_from_edgetrak(fname,roi,n_points):
    """Extract a trace from an Edgetrak trace file
    
    Uses a linear interpolation of the trace to extract evenly-spaced points
    Args:
        fname (str): The path to a trace file.
        roi (ROI): The space accross which to evenly space the points
        n_points (int): The nuber of points to extract
    """
    roi = ROI(roi)
    a = np.fromfile(fname,sep=' ')
    a = a.reshape(-1,6,2)
    a = a.swapaxes(0,1)
    gold_xs = a[trace_in_file,:,0]
    gold_xs = a[trace_in_file,:,1]
    return _trace_interp(gold_xs,gold_ys,roi,n_points)


def trace_from_file(path,roi,n_points,**kwargs):
    """Extract a trace from a trace file

    Uses a linear interpolation of the trace to extract evenly-spaced points
    Args:
        path (str): The path to a trace file.
        roi (ROI): The space accross which to evenly space the points
        n_points (int): The nuber of points to extract
    """
    roi = ROI(roi)
    gold_xs = []
    gold_ys = []
    #TODO regress instead of take first
    with open(path) as f:
        for l in f:
            l = l.split()
            if int(l[0]) > 0:
                gold_xs.append(float(l[1]))
                gold_ys.append(float(l[2]))
    gold_xs = np.array(gold_xs,dtype='float32')
    gold_ys = np.array(gold_ys,dtype='float32')
    return _trace_interp(gold_xs,gold_ys,roi,n_points)

def _trace_interp(xs,ys,roi,n_points):
    """Interpolate a trace

    Args:
        xs (np.array): the original x-values
        ys (np.array): the y-values corresponding to xs
        roi (ROI): The space accross which to evenly space the points
        n_points (int): The nuber of points to extract
    """
    if len(xs) > 0: 
        trace = np.interp(roi.domain(n_points),xs,ys,left=0,right=0)
        trace = trace.reshape((n_points))
        trace[trace==0] = roi.offset[0]
        trace = (trace - roi.offset[0]) / (roi.height)
    else:
        return np.array([0.]*n_points,dtype='float32')
    if trace.sum() > 0 :
        return trace.astype('float32')
    else: 
        return np.array([0.]*n_points,dtype='float32')

def name_from_info(fname,**kwargs):
    """Returns the file name
        
    Returns whatever is in the 'fname' capture.
    
    Parameters
    ----------
        fname (str): The part of the path to return
    """
    if type(fname) is str:
        return np.array(fname)
    else:
        return np.array(fname[0])

def audio_from_file(path,frame,n_samples,__openfiles,fft=False,**kwargs):
    """Extract audio from a multimedia file using pyglet
    
    Supports both plain audio files and video files. 
    
    Parameters
    ----------
    path : str
        The path to the multimedia file

    frame : str(float) 
        The frame number where to extract the audio

    n_samples : int
        The number of audio samples to extract

    fft : bool
        Whether or not to fourrier transform the audio. In practice,
        it's usually simpler to extract the audio raw, and do any 
        transformations afterwards.

    __openfiles : dict
        A place to store opened files, so we don't have to 
        re-open them every time the function is called.
        
    Note
    ----
    This may only have accuracy to about 1/10 sec, depending on file type
    """
    if path not in __openfiles:
        __openfiles[path] = pyglet.media.load(path)
    f = __openfiles[path]
    sample_rate = f.audio_format.sample_rate
    if 'frame_rate' in kwargs:
        frame_rate = kwargs['frame_rate']
    elif f.video_format and f.video_format.frame_rate:
        frame_rate = f.video_format.frame_rate
    else:
        raise ValueError("Could not intuit frame_rate, please specify.")
    frame_rate = float(frame_rate)
    frame = float(frame)
    n_samples = int(n_samples)
    sample_size = f.audio_format.sample_size
    channels = f.audio_format.channels
    t_frame = frame / frame_rate
    t_0 = t_frame - 0.5 * (n_samples/sample_rate)
    f.seek(t_0)
    dtype = 'uint8' if sample_size == 8 else 'int%d'%(sample_size)
    a = np.zeros((channels,0))
    while True:
        d = f.get_audio_data(n_samples)
        if not d: raise Exception("Reached end of file while extracting audio")
        d = np.fromstring(d.data,dtype=dtype)
        d = d.reshape(-1,channels).T
        a = np.append(a,d,axis=1)
        if a.shape[1] >= n_samples: break
    a = a[:n_samples]
    a = a / np.iinfo(dtype).max
    if fft:
        if fft is True:
            a = np.fft.rfft(a).real
        else:
            a = fft(a)
    a = (a / n_samples).astype('float32')
    return a
    

def image_from_video(path,frame,roi,scale,__openfiles,**kwargs):
    """Extract an image frame from a video
    
    Parameters
    ----------
    path : str
        The path to the video file

    frame : str(float)
        The framenumber of the frame to extract

    roi : a :class:ROI, or iterable of int
        The region to extract
    
    scale : float
        How to scale the images

    __openfiles : dict
        A place to store opened files, so we don't have to 
        re-open them every time the function is called.

    frame_rate : float
        Frame rate (in Hertz) of the video. Pyglet can intuit
        this in later versions, but these versions are not 
        on pypi yet.
    """
    if path not in __openfiles:
        __openfiles[path] = pyglet.media.load(path)
    f = __openfiles[path]
    if 'frame_rate' in kwargs:
        frame_rate = kwargs['frame_rate']
    elif f.video_format and f.video_format.frame_rate:
        frame_rate = f.video_format.frame_rate
    else:
        raise ValueError("Could not intuit frame_rate, please specify.")
    roi = ROI(roi)
    roi_scale = roi.scale(scale)
    frame_rate = float(frame_rate)
    frame = float(frame)
    t_frame = frame / frame_rate
    f.seek(t_frame)
    vframe = f.get_next_video_frame()
    a = np.fromstring(vframe.data,dtype='uint8')
    a = a.reshape(vframe.height,vframe.width,-1)
    img = Image.fromarray(a)
    img = img.convert('L')
    img.thumbnail((img.size[0] * scale, img.size[1] * scale))
    img = np.array(img,dtype='float32')
    img = img / 255
    img = np.array(img[roi_scale.slice],dtype='float32')
    img = img.reshape(1,img.shape[0],img.shape[1])
    return img

#TODO rename this
def video_from_file_gen(path,roi,scale,**kwargs):
    """Generator that yields frames from a video, one frame at a time
    
    Parameters
    ----------
    path : str
        The path to the video

    roi : a :class:ROI, or iterable of int
        The region to extract

    scale : float
        How to scale the images

    frame_rate : float
        Frame rate (in Hertz) of the video. Pyglet can intuit
        this in later versions, but these versions are not 
        on pypi yet.
    """
    f = pyglet.media.load(path)
    if 'frame_rate' in kwargs:
        frame_rate = kwargs['frame_rate']
    elif f.video_format and f.video_format.frame_rate:
        frame_rate = f.video_format.frame_rate
    else:
        raise ValueError("Could not intuit frame_rate, please specify.")
    roi = ROI(roi)
    roi_scale = roi.scale(scale)
    frame_rate = float(frame_rate)
    while f.get_next_video_timestamp() < f.duration:
        logging.debug('Timestamp: %f',f.get_next_video_timestamp())
        vframe = f.get_next_video_frame()
        img = np.fromstring(vframe.data,dtype='uint8')
        img = img.reshape(vframe.height,vframe.width,-1)
        img = Image.fromarray(img)
        img = img.convert('L')
        img.thumbnail((img.size[0] * scale, img.size[1] * scale))
        img = np.array(img,dtype='float32')
        img = img / 255
        img = np.array(img[roi_scale.slice],dtype='float32')
        img = img.reshape(1,img.shape[0],img.shape[1])
        yield img

def video_from_file(**kwargs):
    """Return a big tensor with video frames
    
    Same interface as video_from_file_gen"""
    return np.array([img for img in video_from_file_gen(**kwargs)])
