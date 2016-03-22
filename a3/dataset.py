#!/usr/bin/env python3

from __future__ import absolute_import, division

def Dataset(f, *args, **kwargs):
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py and (
        isinstance(f,h5py.File) or isinstance(f,HDF5Dataset) or
        (isinstance(f,str) and f.endswith('hdf5'))):
            return HDF5Dataset(f, *args, **kwargs)
    return NumpyDataset(f, *args, **kwargs)

class AbstractDataset():

    def __init__(self):
        raise NotImplementedError('AbstractDataset is an abstract class, '
            'please use NumpyDataset or HDF5Dataset instead')

    def __getitem__(self,keys):
        try:
            keys = list(keys)
        except TypeError:
            keys = [keys]
        n = keys.pop(0) if keys else slice(None)
        if isinstance(n,str):
            return self.backing[n]
        k = keys.pop(0) if keys else self.keys
        if k == slice(None):
            k = self.keys
        return {kk:self.backing[kk][n][keys if keys else slice(None)]
                for kk in k}

    @property
    def keys(self):
        return self.backing.keys()

    @property
    def N(self):
        N, = {len(self.backing[k]) for k in self.backing} if self.K>0 else 0
        return N

    @property
    def K(self):
        return len(self.keys)

    @property
    def shape(self):
        return {k:self.backing[k].shape[1:] for k in self.keys}

    def split(self, valid=None):
        if not valid:
            valid = int(self.N * .75)
        if hasattr(self,'mode') and self.mode != 'r':
            raise TypeError('Can only split read-only Datasets!')
        return (NumpyDataset(self[valid:]), NumpyDataset(self[:valid]))

class NumpyDataset(AbstractDataset):

    def __init__(self, f=None, *args, **kwargs):
        import numpy
        #TODO: add reading from / writing to .npy file
        if f:
            self.backing = f
        else:
            self.backing = {}

class HDF5Dataset(AbstractDataset):

    def __init__(self, f, mode='r', *args, **kwargs):
        import h5py
        if isinstance(f,str):
            self.backing = h5py.File(f,mode)
        elif isinstance(f,h5py.File):
            self.backing = f
        self.mode = mode
