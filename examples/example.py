#!/usr/bin/env python3

from __future__ import absolute_import

import os
import logging
import h5py

import a3

logging.basicConfig(level=logging.INFO)

# The first step is to create a dataset. This process is mostly abstracted away for you using a3.Dataset objects. Your responsibility is to specify how your data is stored and structured. We'll start by specifying some structure. 

# For my dataset, there are a number of subjects, and for each subject there is a number of ultrasound frames. Importantly, this relationship is heirarchical in that any data that is associated with a subject but not with a frame (for example, an audio recording of the session) is valid for all frames for that subject, whereas information that is associated with a frame number (such as a trace) won't be valid for all subjects. We encode this nesting with the list ['subject','frame']. If, for example, each subject participated 3 times, we might specify a nesting like ['subject','session','frame']. The last element will usually be frame, since that is usually the minimal unit of analysis. Note that these names are arbitrary as long as we are consistant, but using informative names is best practice.
heirarchy = ['subject','frame']

# Next, we need to be able to tell autotres about the types of files we have, the types of data they represent, and what levels of the heirarchy each piece of data should be associated with. This will mostly be accomplished with regular expressions. We create a dict of data types. The keys of the dict represent data type names. Note that we will want one type for each type of data we want,not for each type of file we have. Autotres is perfectly happy pulling more than one type of information from a file. These are again arbitrary labels, but autotres can provide some sensible default behaviors for certain labels. Each dict contains information about that type:
#   'conflict': How to deal with multiple files of the same type appearing for same combination of identifiers. 
#       For example, If I have multiple tracers on my team, then I would expect to have multiple 'trace' files for each combination of 'subject' and 'frame'. In this case, I would like to keep all of the traces for the same image, so that I can do something sensible with it, like interpolate. In this case I will set the value for 'conflict' to 'list'. 
#       Similarly, if I have conducted multiple studies with my dataset, one looking at coronals and one looking at fricatives, I would expect to have multiple copies of the images for coronal fricatives. However, unlike the instance with multiple traces, fricative_frame-00042.png and coronal_frame-00042.png should be identical files. I can use the 'hash' option to specify that I should ignore duplicates as long as they are the same, but should raise an exception if they don't.
#       Finally, there are some situations that I simply expect not to happen. For example, if a single subject is associated with more that one 'audio' file, then perhaps it is most likely that somebdy mislabeled something. In this case, I would not set 'conflict' to anything, and if there is a conflict, autotres will raise an exception automatically
#   'regex': How to associate each file with a combination of heirachical levels.
#       This is a regular expression that will match a filename in the dataset. We use the (?P<label>...) syntax to capture parts of the filename that are informative. Specifically, we need to be able to infer all the relevent heirarchical information from the file name. Note that the regex is matched to the entire pathname, relative to whatever path we give it (see below). Here, we have also used the (?x) flag to allow us to break the regex over multiple lines, and to include comments.
types = {
    'trace': {
        'conflict': 'list',
        'regex': r"""(?x)
            %(sep)s                 # from (the last) file separater
            (?P<subject>\d\d[a-c])  # we expect a 'subject' ID of the form: 42a
            [^%(sep)s]+             # and a bunch of junk, (but not a file separator)
            frame-(?P<frame>\d+)    # then the literal word frame, followed by a 'frame' number
            \.(?:png|jpg)           # an image file extension
            \.(?P<tracer>\w+)       # the tracer's initials
            \.traced\.txt$          # and our trace files end with .txt
            """%{'sep':os.sep},
        },
    'image': {
        'conflict': 'hash'
        'regex': r"""(?x)
            %(sep)s                 # like trace files
            (?P<subject>\d\d[a-c])  # here's our 'subject'
            [^%(sep)s]+             # bunch of junk
            frame-(?P<frame>\d+)    # frame number
            \.(?P<ext>png|jpg)$     # file extension
            """%{'sep':os.sep},
        },
    'name': {
	    'conflict':'list',
        'regex': r"""(?x)                   # this will give us the file name of the image
            (?P<subject>\d\d[a-c])_[-0-9]+  # 'subject' id
            .*                              # 
            %(sep)s                         # (last) file separator
            (?P<fname> [^%(sep)s]+          # we want the whole file name
            frame-(?P<frame>\d+)            # including the 'frame' number, with a nested (?P<>) 
            \.(?:png|jpg))$                 # up to the end of the filename
            """%{'sep':os.sep},
        }
    }

# We'll save our dataset as SG.hdf5, but only if it doesn't already exist
if not os.path.isfile('SG.hdf5'):
    # We will not setup our dataset. The roi, n_points, and scale kwargs are required for the default data extraction callbacks (see a3/dataset.py documentation). Custom callbacks can be provided by putting a callable in the ds.callbacks dict. These should return a numpy array of type float32 
    ds = a3.Dataset('SG.hdf5',roi=(140.,320.,250.,580.),n_points=32,scale=.1)
    # We will scan our data directory, ./SG/ using our types and our heirarchy. You can scan multiple directories if you need to, possibly with different type definitions, but watch out for file conflicts! Your heirarchy should be the same accross calls to scan_directory.
    ds.scan_directory('./SG/',types,heirarchy)
    # At this point, you can inspect what data sources you have by looking at the ds.sources dict. Watch out, this dict can get very large, so don't try printing to stdout unless you've got a while.
    # Once you have your data sources figured out, you can extract that data with ds.read_sources(). The arg here is a set-like object with all of the data types you need. 
    ds.read_sources({'trace','image','name'})

# The rest is easy. Construct an autotracer from your new dataset. Specifying None for the validation set sets aside part of your training data as validation data (no guarantees about randomness). Make sure you use the same ROI as above, or at least the same size.
a = a3.Autotracer('SG.hdf5',None,(140.,320.,250.,580.),'example.a3.json')
# Train on the dataset 10 times. In reality, training will require thousands of runs through the dataset. Mnibatch size can be controlled with the 'minibatch' kwarg, which defaults to 512.
a.train(10)
# Save your weights! Note that this doesn't contain any information about the layout of the NNet -- that's still in the works. To change layouts, change the code in a3.Autotrace.__init_layers()
a.save('example.a3.npy')
# get the traces for your dataset!
with h5py.File('SG.hdf5','r') as h:
    a.trace(h['image'],'traces.json',h['name'],'autotest','042')
