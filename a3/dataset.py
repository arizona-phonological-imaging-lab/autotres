#!/usr/bin/env python3

from __future__ import absolute_import, division

def dataset(f, *args, **kwargs):
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py and (
        isinstance(f,h5py.File) or isinstance(f,HDF5Dataset) or
        (isinstance(f,basestring) and f.endswith('hdf5'))):
            return HDF5Dataset(f, *args, **kwargs)
    return NumpyDataset(f, *args, **kwargs)

class Dataset():

    def __init__(self):
        raise NotImplementedError('Dataset is an abstract class, '
            'please use NumpyDataset or HDF5Dataset instead')

    def __getitem__(self,*keys):
        n = keys.pop(0) if keys else slice(None)
        k = keys.pop(0) if keys else self.keys
        if k == slice(None):
            k = self.keys
        if k == None:
            return {kk:self.backend[kk][n][keys if keys else slice(None)]
                    for kk in k}
        else:
            return [self.backing[kk][n][keys if keys else slice(None)]
                    for kk in k]

class NumpyDataset(Dataset):

    def __init__(self, f=None, *args, **kwargs):
        import numpy
        #TODO: add reading from / writing to .npy file
        if f:
            self.backing = f
        else:
            self.backing = {}
        self.N, = {len(self.backing[k]) for k in self.backing}

class HDF5Dataset(Dataset):

    def __init__(self, f, *args, **kwargs):
        import h5py

