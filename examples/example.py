#!/usr/bin/env python3

from __future__ import absolute_import

import os
import logging
import h5py

import a3

types = {
    'trace': {
        'conflict': 'list',
        'regex': r"""(?x)
            %(sep)s
            (?P<subject>\d\d[a-c])
            [^%(sep)s]+
            frame-(?P<frame>\d+)
            \.(?:png|jpg)
            \.(?P<tracer>\w+)
            \.traced\.txt$
            """%{'sep':os.sep},
        },
    'image': {
        'regex': r"""(?x)
            %(sep)s
            (?P<subject>\d\d[a-c])
            [^%(sep)s]+
            frame-(?P<frame>\d+)
            \.(?P<ext>png|jpg)$
            """%{'sep':os.sep},
        'conflict': 'hash'
        },
    'name': {
	'conflict':'list',
        'regex': r"""(?x)
            (?P<subject>\d\d[a-c])_[-0-9]+
            .*
            %(sep)s
            (?P<fname> [^%(sep)s]+
            frame-(?P<frame>\d+)
            \.(?:png|jpg))$
            """%{'sep':os.sep},
        }
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    roi = a3.ROI(140.,320.,250.,580.)
    if not os.path.isfile('test.hdf5'):
        ds = a3.Dataset('SG.hdf5',roi=roi,n_points=32,scale=.1)
        ds.scan_directory('./SG/',types,['subject','frame'])
        ds.read_sources(types.keys())
    a = a3.Autotracer('SG.hdf5',None,roi)
    a.train()
    a.save('example.a3.npy')
    with h5py.File('SG.hdf5','r') as h:
        a.trace(a.X_valid,'traces.json',h['name'],'autotest','042')
