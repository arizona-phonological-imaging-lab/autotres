#!/usr/bin/env python3

import h5py
import os
import logging
import re
import hashlib
import numpy as np
from PIL import Image
try:
    from .errors import ConflictError
except:
    from errors import ConflictError
try:
    from .roi import ROI
except:
    from roi import ROI

try: # for python2/3 compatibility
    unicode
except NameError:
    unicode=str

class Dataset(object):

    def __init__(self,backing='dict',**kwargs):
        self.sources = {}
        self.data = None
        self.callbacks = {
            'image': _image_from_file,
            'trace': _trace_from_file,
            'name' : _name_from_info,
            }
        if 'dname' in kwargs:
            self.scan_directory(kwargs['dname'],**kwargs)
        if backing.endswith('hdf5'):
            self.__backing = h5py.File(backing)
        else:
            self.__backing = {}
        self.settings = kwargs

    def __getitem__(self,key):
        return self.__backing[key]

    def __setitem__(self,key,value):
        if key in self:
            del self[key]
        if type(value) == tuple:
            maxshape = (None,) + value[1:]
            if type(self.__backing) == dict:
                self.__backing[key] = np.ndarray(
                    value,
                    dtype='<U15')
            elif type(self.__backing) == h5py.File:
                dtype = (h5py.special_dtype(vlen=unicode)
                     if key.lower() in ['id','name'] else 'f32' )
                self.__backing.create_dataset(
                    key,
                    shape=value,
                    maxshape=maxshape,
                    dtype=dtype)
        elif type(value) == np.ndarray:
            if type(self.__backing) == dict:
                self.__backing[key] = value
            elif type(self.__backing) == h5py.File:
                dtype = (h5py.special_dtype(vlen=unicode)
                     if key.lower() in ['id','name'] else 'f32' )
                maxshape = (None,) + value.shape[1:]
                self.__backing.create_dataset(
                    key,
                    shape=value.shape,
                    maxshape=maxshape,
                    dtype=dtype)
                self.__backing[key][:] = value
        else: raise TypeError

    def __delitem__(self,key):
        del self.__backing[key]

    def __contains__(self,item):
        return item in self.__backing
    
    def __exit__(self):
        try:
            self.__backing.close()
        except AttributeError:
            pass

    def scan_directory(self,d,types,keys,sep=':',report_every=1000):
        N_matches = 0
        for dirpath,__,filenames in os.walk(d):
          logging.debug('entering %s...',dirpath)
          for filename in filenames:
            fullpath = os.sep.join((dirpath,filename))
            for stype in types: #"source type"
                match = re.search(types[stype]['regex'],fullpath)
                if match:
                    N_matches += 1
                    if not N_matches % report_every:
                        logging.debug('matched %d files',N_matches)
                    match_keys = match.groupdict()
                    node = self.sources
                    ID = []
                    for key in keys:
                        if key in match_keys:
                            if match_keys[key] not in node:
                                node[match_keys[key]] = {}
                            node = node[match_keys[key]]
                            ID.append(match_keys[key])
                        else:
                            break
                    if stype not in node:
                        node[stype] = {}
                    node = node[stype] 
                    ID = sep.join(ID)
                    #insert
                    if 'conflict' not in types[stype]:
                        if 'path' in node:
                            logging.error(
                                '%s conflicts with %s for item %s, type %s',
                                node['path'],fullpath,ID,stype)
                            raise ConflictError(ID,
                                                node['path'],fullpath)
                        else:
                            node['path'] = fullpath
                            if any((k in node for k in match_keys)):
                                raise Exception
                            node.update(match_keys)
                            if 'id' not in node: node['id'] = ID
                    elif types[stype]['conflict'] == 'hash':
                        md5 = _hash_md5(fullpath)
                        if 'hash' in node and md5 != node['hash']:
                            logging.error(
                                ('%s(md5sum:%s) hash-conflicts with '
                                 'item %s(md5sum:%s) for item %s'),
                                node['path'],node['hash'],
                                fullpath,md5,ID)
                            raise ConflictError(ID,
                                                node['path'],fullpath)
                        node['path'] = fullpath
                        node['hash'] = md5
                        if any((k in node and node[k] != match_keys[k] 
                                for k in match_keys)):
                            raise Exception
                        node.update(match_keys)
                        if 'id' not in node: node['id'] = ID
                    elif types[stype]['conflict'] == 'list':
                        if 'path' not in node:
                            node['path'] = []
                        node['path'].append(fullpath)
                        for k in match_keys:
                            if k not in node:
                                node[k] = []
                            node[k].append(match_keys[k])
                        if 'id' not in node: node['id'] = ID
                    else:
                        raise NotImplementedError

    def read_sources(self,types):
        self.dat = self.__read_sources(self.sources,types,{})
        dat =self.dat
        ids = {d[k]['id'] for d in dat for k in d}
        N = len(ids)
        self['id'] = np.array(list(ids))
        for d in dat:
            assert 1==len({d[k]['id'] for k in d})
            for k in types:
                val = self.callbacks[k](**dict(d[k],**self.settings))
                if k not in self:
                    self[k] = (N,) + val.shape
                i = np.where(self['id'][:] == d[k]['id'])[0][0]
                self[k][i] = val

    def __read_sources(self,node,types,context):
        type_1 = {k:node[k] for k in node if k not in types}
        type_2 = {k:node[k] for k in node if k in types}
        d = dict(context,**type_2)
        if all((x in d for x in types)):
            return (d,)
        else:
            return [x for k in type_1 
                    for x in self.__read_sources(type_1[k],types,d)]

def _hash_md5(fname,buff=1024):
    with open(fname,'rb') as f:
        m = hashlib.md5()
        for b in __buffered_read(f,buff):
            m.update(b)
        return m.hexdigest()

def __buffered_read(f,buff):
    dat = True
    while dat:
        dat = f.read(buff)
        if dat: yield dat

def _image_from_file(path,roi,scale,**kwargs):
    """Extract a porperly scaled section of an image

    Args:
        path (str): The path to an image
        roi (ROI): The part of the image to extract
        scale
    """
    roi = ROI(roi)
    roi_scale = roi.scale(scale)
    img = Image.open(path)
    img = img.convert('L')
    img.thumbnail((img.size[0] * scale, img.size[1] * scale))
    img = np.array(img,dtype='float32')
    img = img / 255
    img = np.array(img[roi_scale.slice],dtype='float32')
    img = img.reshape(1,img.shape[0],img.shape[1])
    return img

def _trace_from_file(path,roi,n_points,**kwargs):
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
    with open(path[0]) as f:
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

def _name_from_info(fname,**kwargs):
    try:
        return np.array(fname[0])
    except TypeError:
        return np.array(fname)

_types = {
    'trace': {
        'regex': r'(?P<study>\d+\w+)_(?P<frame>\d+)\.(?:jpg|png)\.(?P<tracer>\w+)\.traced\.txt$',
        'conflict': 'list'
        },
    'image': {
        'regex': r'(?P<study>\d+\w+)_(?P<frame>\d+)\.(?P<ext>jpg|png)$',
        'conflict': 'hash'
        },
    'name': {
        'regex': r'(?P<fname>(?P<study>\d+\w+)_(?P<frame>\d+)\.(?P<ext>jpg|png))$',
    }
    }
if __name__=='__main__':
    roi = (140,320,250,580)
    n_points = 32
    keys = ['study','frame']
    types = _types
    ds = Dataset(backing="test.hdf5",roi=roi,n_points=n_points)
    ds.scan_directory('./test_data',types,keys)
    ds.read_sources(types.keys())
