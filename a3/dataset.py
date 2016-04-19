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
        from . import lib
        self.CALLBACKS = {
            'image': lib.image_from_file,
            'trace': lib.trace_from_file,
            'name' : lib.name_from_info,
            'audio': lib.audio_from_file,
            }
        if type(self) == AbstractDataset:
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
        N, = {len(self.backing[k]) for k in self.backing} if self.K>0 else (0,)
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

    def add_data(self,tree,stypes):
        import logging
        logging.info("adding data from %s",tree)
        N = 0
        for datum in tree(stypes.keys()):
            N += 1
            if not N%1000:
                logging.info('processed %d datapoints',N)
            i = datum.get('ID',self.N)
            for k, stype in stypes.items():
                callback = (stype['callback']
                    if 'callback' in stype
                    else self.CALLBACKS[k])
                dat = callback(datum[k],**stype.get('const',{}))
                self[i,k] = dat

class NumpyDataset(AbstractDataset):

    def __init__(self, f=None, *args, **kwargs):
        super().__init__()
        import numpy
        #TODO: add reading from / writing to .npy file
        if f:
            self.backing = f
        else:
            self.backing = {}

class HDF5Dataset(AbstractDataset):

    def __init__(self, f, mode='r', *args, **kwargs):
        super().__init__()
        import h5py
        if isinstance(f,str):
            self.backing = h5py.File(f,mode)
        elif isinstance(f,h5py.File):
            self.backing = f
        self.mode = mode
    
    def __setitem__(self,keys,val):
        assert self.mode.count('w')
        i, k, *keys = keys
        if k not in self.backing:
            if val.dtype.char=='U':
                import h5py
                dtype = h5py.special_dtype(vlen=str)
            else:
                dtype = val.dtype
            self.backing.create_dataset(
                name=k,
                shape=(self.N,)+val.shape,
                dtype=dtype,
                chunks=(512,)+val.shape,
                maxshape=(None,)+val.shape,
                compression='gzip',
                shuffle=True,)
        if i >= self.N:
            for kk in self.keys:
                self.backing[kk].resize(i+1,axis=0)
        self.backing[k][i,...] = val

class DataSourceTree():

    def __init__(self,keys):
        self.keys = keys
        self.inherit = {}
        self.subtree = {}

    def __setitem__(self,ID,val):
        if ID: # we have more to traverse
            k = ID[0]
            if k not in self.subtree:
                self.subtree[k] = DataSourceTree(self.keys[1:])
            self.subtree[k][ID[1:]] = val
        else: # put it here
            #TODO: add conflict types
            self.inherit.update(val)

    def scan_dir(self,stypes,root):
        import os
        import re
        import logging
        logging.info("scanning directory %s",root)
        flag = 0
        for dirpath, __, f in os.walk(root,followlinks=True):
          for filename in f:
            fullpath = os.sep.join([dirpath,filename])
            if not flag % 1000:
                flag = 1
            else:
                flag += 1
            for stype in stypes:
                match = re.search(stypes[stype]['regex'],fullpath)
                if match:
                    matches = match.groupdict()
                    ID = []
                    for k in self.keys:
                        if k in matches:
                            ID += [matches[k]]
                        else:
                            break
                    self[ID] = {stype:fullpath}

    def __iter__(self):
        return self()

    def __call__(self,stypes=None):
        if stypes:
            for x in self._all_nodes():
                if all(t in x for t in stypes):
                    yield x
        else:
            for x in self._all_nodes():
                yield x

    def _all_nodes(self, inherit=None):
        if inherit is None:
            from collections import deque
            inherit = deque()
        inherit.appendleft(self.inherit)
        if self.keys:
            for st in self.subtree.values():
                for x in st._all_nodes(inherit):
                    yield x
        else:
            r = {}
            for d in inherit:
                r.update(d)
            yield r
        inherit.popleft()
