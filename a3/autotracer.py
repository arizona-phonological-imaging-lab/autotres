#!/usr/bin/env python3

from __future__ import absolute_import, division

from .roi import ROI
from .errors import ShapeError
from .constants import _version
from lasagne.regularization import regularize_layer_params, regularize_layer_params_weighted, l2, l1
from .struct import LossRecord
from .utils import get_path, Config
import json
import logging
import sys

import numpy as np
import h5py
import theano
import theano.tensor as T
import lasagne

class Autotracer(object):
    """Automatically traces tongues in Ultrasound images.

    Attributes:
        roi : :class:`ROI`
            Where the ultrasound images the data represent.
        train_data : :class:`Dataset`
            All of the data to be used in training, including both
            the preditors (usually images) and predictions (traces)
        valid_data : :class:`Dataset`
            All of the data to be used for validation.
            Like `train_data`, includes predictors and predictions
    """

    # TODO: alternative constructor using data from Config instance
    # TODO: incorporate network json into config
    def __init__(self, net_json, roi, train=None, valid=None, config=None):
        """

        Args:
            net_json : string
                The location of a network definition file
            roi : :class:`ROI` 
                The location of the data within an image
                This is mostly used for tracing with a trained
                network, to output the correct (x,y) coordinates
            train : :class:`Dataset` or string
                The training dataset (or location thereof)
                This gets loaded as `train_data`
            valid : :class:`Dataset` or string
                The validation dataset (or location thereof)
                This gets loaded as `valid_data`
            config : string
                The location of a network configuration file
        """
        self.__nonlinearities = {
            'relu' : lasagne.nonlinearities.rectify}
        self._layers = {}
        self.outputLayers = []
        self.config = Config(config) if config else config

        if train:
            # clean up paths
            train = get_path(train)
            valid = get_path(valid) if valid else valid
            self.loadDataset(train, valid)
        self.roi = ROI(roi)
        self.__init_layers(net_json)

    @property
    def Xshape(self):
        """A dict mapping predictors to their expected shapes"""
        return {l.name:l.shape[1:] for l in self.layer_in}

    @property
    def yshape(self):
        """The shape of the output"""
        return self.layer_out.shape[1:]

    @property
    def predictors(self):
        """A list of the predictors (inputs) for the net"""
        return [l.name for l in self.layer_in]

    def loadDataset(self,train,valid=None):
        """Load a training and validation dataset from hdf5 databases

        Args:
            train (string): the location of a hdf5 dataset for training
                this gets loaded as X_train and y_train
            valid (string): the location of a hdf5 dataset for validation
                this gets loaded as X_valid and y_valid. A value of None
                means that the train dataset should be split to create it

        Raises:
            ShapeError: if train and valid have incompatible shape
                Xshape and yshape are set to the shape of valid
        """
        # clean up paths
        train = get_path(train)
        valid = get_path(valid) if valid else valid
        logging.debug('loadHDF5(%s,%s)' % (train,valid))
        self.train_data = Dataset(train)
        if valid:
            self.valid_data = Dataset(valid)
        else:
            # split the training data into a training set and a validation set.
            self.train_data, self.valid_data = self.train_data.split(valid)
        mismatch = False
        mismatches = {k for k in self.train_data.keys() & self.valid_data.keys()
               if self.train_data[k].shape[1:] != self.valid_data[k].shape[1:]}
        if len(mismatches) > 0:
            for k in mismatches:
                logging.warn('Training and validation data have '
                    'different shape for "%s"',k)
            raise ShapeError({k:(train_data[k].shape,valid_data[k].shape) 
                              for k in mismatches})

    def __init_layers(self,jfile,encoding='IBM500'):
        """Create the architecture of the MLP

        Args:
            jfile (string): Location of a json file specifying the 
                desired architecture for the network. 
                See examples/ for example files
            encoding (string): A python built-in encoding for loading 
                the binary bytestring from JSON. Defaults to EBCDIC.

        Raises:
            ShapeError: if input and/or output dimensionality are unset
        """
        with open(jfile,'rt') as f:
            d = json.load(f)
        self.layer_in = [] # will be filled by __init_layers_file_recursive
        for k in d:
            self.__init_layers_file_recursive(d,k,encoding)
        #this enforced single output; TODO don't
        self.layer_out, = self.outputLayers

    def __init_layers_file_recursive(self,d,cur,enc):
        """Recursively traverse the architecture definition in d

        Args:
            cur (string): The key of the current layer. d[cur] should be 
                a dict describing a layer of the network
            enc (string): A python built-in encoding for loading 
                the binary bytestring from JSON. Defaults to EBCDIC.
        """
        if cur in self._layers:
            return self._layers[cur]
        l_type = d[cur]['type']
        if l_type == 'dense':
            l_input = self.__init_layers_file_recursive(d,d[cur]['input'],enc)
            l_nl = (self.__nonlinearities[d[cur]['nonlinearity']] 
                if 'nonlinearity' in d[cur] 
                else lasagne.nonlinearities.rectify)
            dtype = (d[cur]['dtype'] if 'dtype' in d[cur] 
                else theano.config.floatX)
            l_num_units = int(d[cur]['num_units'])
            l_W = (np.fromstring(d[cur]['W'].encode(enc),dtype).reshape((-1,l_num_units)) 
                if 'W' in d[cur] else lasagne.init.GlorotUniform())
            l_b = (np.fromstring(d[cur]['b'].encode(enc),dtype) 
                if 'b' in d[cur] else lasagne.init.Constant(0.))
            l = lasagne.layers.DenseLayer(
                l_input,
                nonlinearity = l_nl,
                num_units = l_num_units,
                W = l_W,
                b = l_b,
                name = cur)
        elif l_type == 'input':
            if 'shape' in d[cur]:
                l_shape = tuple(d[cur]['shape'])
            elif hasattr(self,'train_data'):
                l_shape = self.train_data[cur].shape[1:]
            else:
                raise RuntimeError('Cannot guess shape for input "%s"' % (cur,))
            l_shape = (None,) + l_shape
            l = lasagne.layers.InputLayer(
                shape = l_shape,
                name = cur)
            if cur not in {l.name for l in self.layer_in}:
                self.layer_in.append(l)
        elif l_type == 'output':
            if 'shape' in d[cur]:
                l_shape = tuple(d[cur]['shape'])
            elif hasattr(self,'y_train'):
                l_shape = self.y_train.shape[1:]
            else:
                raise RuntimeError('Cannot guess shape for outputput "%s"' % (cur,))
            l_input = self.__init_layers_file_recursive(d,d[cur]['input'],enc)
            l = Autotracer.OutputLayer(
                l_input,
                l_shape,
                config = self.config,
                name = cur)
            self.outputLayers.append(l)
        elif l_type == 'dropout':
            l_input = self.__init_layers_file_recursive(d,d[cur]['input'],enc)
            l_p = float(d[cur]['p']) if 'p' in d[cur] else 0.5
            l = lasagne.layers.DropoutLayer(
                l_input,
                p = l_p,
                name = cur)
        elif l_type == 'concat':
            l_inputs = [self.__init_layers_file_recursive(d,k,enc)
                for k in d[cur]['inputs']]
            l_axis = d[cur]['axis'] if 'axis' in d[cur] else 1
            l = lasagne.layers.ConcatLayer(
                l_inputs,
                axis = l_axis,
                name = cur)
        elif l_type == 'conv':
            l_input = self.__init_layers_file_recursive(d,d[cur]['input'],enc)
            l_num_filters = int(d[cur]['num_filters'])
            l_filter_size = d[cur]['filter_size']
            l_stride = d[cur].get('stride',(1,1))
            l_pad = d[cur].get('pad',0)
            dtype = d[cur].get('dtype',theano.config.floatX)
            if 'W' in d[cur]:
                l_W = np.fromstring(d[cur]['W'].encode(enc),dtype)
                try:
                    l_W = l_W.reshape((l_num_filters,-1,int(l_filter_size),int(l_filter_size)))
                except TypeError:
                    l_W = l_W.reshape((l_num_filters,-1,int(l_filter_size[0]),int(l_filter_size[0])))
            else:
                l_W = lasagne.init.GlorotUniform()
            l_b = (np.fromstring(d[cur]['b'].encode(enc),dtype)
                if 'b' in d[cur] else lasagne.init.Constant(0.))
            l_nl = (self.__nonlinearities[d[cur]['nonlinearity']]
                if 'nonlinearity' in d[cur]
                else lasagne.nonlinearities.rectify)
            l = lasagne.layers.Conv2DLayer(
                l_input,
                l_num_filters,
                l_filter_size,
                stride = l_stride,
                pad = l_pad,
                W = l_W,
                b = l_b,
                nonlinearity = l_nl,
                name = cur)
        else:
            raise NotImplementedError("Cannot (yet) load %s layers."%(l_type))
        l._l1_reg = float(d[cur].get('l1_regularization',
                          self.config.l1_regularization if self.config else 0))
        l._l2_reg = float(d[cur].get('l2_regularization',
                          self.config.l2_regularization if self.config else 0))
        if isinstance(l,Autotracer.OutputLayer):
            self._layers[cur+"__dense"] = l.l_dense
            l = l.l_reshape
        self._layers[cur] = l
        return l

    def save(self,fname,save_params=True,encoding='IBM500'):
        """Save the current network to a file
 
        Args:
            fname (string): Where to save the network.
            save_params (bool): Whether or not to save netowrk weights
            encoding (string): A python built-in encoding for saving
                the binary bytestring as a JSON string. Defaults to EBCDIC.
        """
        d = {}
        self.__save_recursive(d,self.layer_out,save_params,encoding)
        with open(fname,'wt') as f:
            json.dump(d,f)

    def __save_recursive(self,d,layer,sp,enc):
        """Recursivey builds a representation of the network

        Args:
            d (dict): A dict of dicts, representing the network.
                Each call adds an element to d.
            layer (string): The name of the current layer
            sp (boolean): Whether to save the network weights
            enc (string): Encoding to use for saving the binary 
                bytestring as JSON.

        Returns:
            (string): The name of the layer that has just been added
        """
        i = layer.name
        if i in d:
            return
        t = {}
        if type(layer) == lasagne.layers.DenseLayer:
            t['type'] = 'dense'
            t['input'] = self.__save_recursive(d,layer.input_layer,sp,enc)
            t['nonlinearity'], = [nl for nl in self.__nonlinearities 
                   if self.__nonlinearities[nl] == layer.nonlinearity]
            if sp:
                t['dtype'] = layer.W.get_value().dtype.str
                t['W'] = layer.W.get_value().tobytes().decode(enc)
                t['b'] = layer.b.get_value().tobytes().decode(enc)
            t['num_units'] = layer.num_units
        elif type(layer) == lasagne.layers.noise.DropoutLayer:
            t['type'] = 'dropout'
            t['input'] = self.__save_recursive(d,layer.input_layer,sp,enc)
            t['p'] = layer.p
        elif type(layer) == lasagne.layers.merge.ConcatLayer:
            t['type'] = 'concat'
            t['inputs'] = [self.__save_recursive(d,l,sp,enc) 
                           for l in layer.input_layers]
            t['axis'] = layer.axis
        elif type(layer) == lasagne.layers.conv.Conv2DLayer:
            t['type'] = 'conv'
            t['input'] = self.__save_recursive(d,layer.input_layer,sp,enc)
            t['num_filters'] = layer.num_filters
            t['filter_size'] = layer.filter_size
            t['stride'] = layer.stride
            t['pad'] = layer.pad
            t['nonlinearity'], = [nl for nl in self.__nonlinearities
                   if self.__nonlinearities[nl] == layer.nonlinearity]
            if sp:
                t['dtype'] = layer.W.get_value().dtype.str
                t['W'] = layer.W.get_value().tobytes().decode(enc)
                t['b'] = layer.b.get_value().tobytes().decode(enc)
        elif type(layer) == lasagne.layers.input.InputLayer:
            t['type'] = 'input'
            t['shape'] = list(layer.shape[1:])
        d[i] = t
        return i

    def train_batch(self, *args):
        """Train on a minibatch

        Wrapper for _train_fn()

        Args:
            X (tensor of float32): Minibatch from the training images
            y (tensor of float32): The corresponding traces
        """
        return self._train_fn(*args)

    def valid_batch(self, *args):
        """Validates the network on a (mini)batch

        Wrapper for _valid_fn()

        Args:
            X (tensor of float32): Minibatch from the validation images
            y (tensor of float32): The corresponding traces
        """
        return self._valid_fn(*args)

    def trace(self, X, jfile=None, names=None, project_id=None, subject_id=None):
        """Trace a batch of images using the MLP

        Can be used programmatically to get a numpy array of traces,
        or a json file for use with the APIL webapp.
        Args:
            X (tensor of float32): image to be traced
                should be properly scaled to [0,1] and the roi.
            jfile (string, optional): location to save json traces
                If None, then no json trace is created
                The rest of the args are required if jfile is truthy
            names (list of str, semi-optional): filenames for each trace
                Used to associate traces in json with files
            project_id (json-ible object): the project id
                This is purely user-defined. How you identify projects.
                Suggestions include strings or numbers
            subject_id (json-ible object): the subject id
                This is also user-defined. How you identify subjects.
                Suggestions again include strings and numbers
        Returns:
            numpy.array of float32: traces for each image
                The traces will be scaled up to the scale of the image,
                rather than on the scale required for input.
        """
        t, = self._trace_fn(*[X[l.name] for l in self.layer_in])
        if jfile:
            # expand path
            jfile = get_path(jfile)
            domain = self.roi.domain(t.shape[1])
            js = { 'roi'     : self.roi.json(),
                'tracer-id'  : 'autotrace_%d.%d.%d'%_version,
                'project-id' : project_id,
                'subject-id' : subject_id}
            js['trace-data'] = {names[i]: [{'x': domain[j], 'y': float(t[i,j])}
                for j in range(len(domain)) if
                float(t[i,j]) != self.roi.offset[1]] for i in range(len(t))}
            with open(jfile,'w') as f:
                json.dump(js,f)
        return t

    def trace_vid(self, path, scale, outdir=False, jfile=False, **kwargs):
        import pyglet
        from PIL import Image, ImageDraw
        import os
        f = pyglet.media.load(path)
        if 'frame_rate' in kwargs:
            frame_rate = kwargs['frame_rate']
        elif f.video_format and f.video_format.frame_rate:
            frame_rate = f.video_format.frame_rate
        else:
            raise ValueError("Could not intuit frame_rate, please specify.")
        if 'roi' in kwargs:
            roi = kwargs['roi']
        else:
            roi = self.roi
        roi_scale = roi.scale(scale)
        domain = roi.domain(self.layer_out.output_shape[1])
        frame_rate = float(frame_rate)
        frame = -1
        ts = []
        while f.get_next_video_timestamp() < f.duration:
            frame += 1
            vframe = f.get_next_video_frame()
            img = np.fromstring(vframe.data,dtype='uint8')
            img = img.reshape(vframe.height,vframe.width,-1)
            img = Image.fromarray(img)
            fullimg = img
            img = img.convert('L')
            img.thumbnail((img.size[0] * scale, img.size[1] * scale))
            img = np.array(img,dtype='float32')
            img = img / 255
            img = np.array(img[roi_scale.slice],dtype='float32')
            img = img.reshape(1,1,img.shape[0],img.shape[1])
            t, = self._trace_fn(img)
            ts.append(t)
            if outdir:
                draw = ImageDraw.Draw(fullimg)
                points = [xy for xy in zip(domain,t[0])]
                draw.line(points,fill=128,width=3)
                fullimg.save(os.sep.join([outdir,'frame-%07d.png'%frame]))
        ts=np.array(ts)
        if jfile:
            if all((x in kwargs for x in ('project_id','subject_id'))):
                project_id = kwargs['project_id']
                subject_id = kwargs['project_id']
            else:
                raise ValueError('must specify project_id and subject_id')
            if 'name' in kwargs:
                name = kwargs['name']
            else:
                name = 'frame-%07(frame)d.png'
            jfile = get_path(jfile)
            js = { 'roi'     : self.roi.json(),
                'tracer-id'  : 'autotrace_%d.%d.%d'%_version,
                'project-id' : project_id,
                'subject-id' : subject_id}
            js['trace-data'] = {
                names[i]: [{'x': domain[j], 'y': float(t[i,j])}
                for j in range(len(domain)) if
                float(t[i,j]) != self.roi.offset[1]] for i in range(len(ts))}
            with open(jfile,'w') as f:
                json.dump(js,f)
        return ts

    def test(self,prediction='trace',test=None,other=None,loss=lasagne.objectives.squared_error, inf=10000):
        this, = [l for l in self.outputLayers if l.name == prediction]
        if isinstance(other,Autotracer):
            other, = [l for l in other.outputLayers if l.name == prediction]
        if not other:
            other = _FakeOutputLayer(self.y_train)
        if test:
            with h5py.File(test) as h:
                gold = h[prediction]
                test_data = {k:np.array(h[k]) for k in gold.keys()}
        else:
            gold = self.y_valid
            test_data = self.valid_data
        return this.test(test_data, other, gold, loss, inf)

    def graph(self,path=None,format='svg',rankdir='TB'):
        import pygraphviz
        g = pygraphviz.AGraph(directed=True,rankdir=rankdir)
        self.__graph_recursive(g,self.layer_out)
        g.layout('dot')
        r = g.draw(path=path,format=format)
        return r

    def __graph_recursive(self,graph,layer):
        i = hex(id(layer)) 
        if graph.has_node(i):
            return i
        if type(layer) == lasagne.layers.DenseLayer:
            shape = '*'.join([str(dim) for dim in layer.output_shape if dim and dim > 1])
            attrs = {
                'shape':'rectangle',
                'label':'%s:\n%s'%(layer.name,shape)}
            layers = [layer.input_layer]
        elif type(layer) == lasagne.layers.noise.DropoutLayer:
            attrs = {
                'label':'%s:\np=%g'%(layer.name,layer.p)}
            layers = [layer.input_layer]
        elif type(layer) == lasagne.layers.merge.ConcatLayer:
            layers = layer.input_layers
            attrs = {
                'shape':'house',
                'label':'%s:\n%s-way'%(layer.name,len(layers))}
            if len(layers) == 1:
                return self.__graph_recursive(graph,layers[0])
        elif type(layer) == lasagne.layers.input.InputLayer:
            shape = '*'.join([str(dim) for dim in layer.output_shape if dim and dim > 1])
            attrs = {
                'shape':'trapezium',
                'label':'%s:\n%s'%(layer.name,shape),}
            layers = []
        graph.add_node(i,**attrs)
        for l in layers:
            j = self.__graph_recursive(graph,l)
            graph.add_edge(j,i)
        return i

    def train(self,num_epochs=2500,batch_size=512,best=False):
        """Train the MLP using minibatches

        Args:
            num_epochs (int): Number of times to run through the
                training set during each epoch.
            batch_size (int): Number of images to calculate updates on
        """
        logging.info('Training')
        if not all((hasattr(self,x) for x in 
                   ('train_data','valid_data'))):
            logging.warning('Cannot train without training data!')
            return False
        # keep track of (epoch + 1, train_loss, valid_loss)
        self.loss_record = LossRecord()
        if best:
            best_loss = sys.float_info.max
            best_params = np.array(lasagne.layers.get_all_param_values(self.layer_out))
        try:
            for outputLayer in self.outputLayers:
                for epoch_num in range(num_epochs):
                    num_batches_train = math.ceil(len(self.train_data) / batch_size)
                    train_losses = []
                    for batch_num in range(num_batches_train):
                        batch_slice = slice(batch_size * batch_num,
                                            batch_size * (batch_num +1))
                        X_batch = self.train_data[batch_slice][outputLayer.predictors]
                        y_batch = self.train_data[batch_slice][outputLayer.predictions]
                        loss = outputLayer.train(*(X_batch + y_batch))
                        train_losses.append(loss)
                    train_loss = np.mean(train_losses)
                    X_valid = self.valid_data[:][outputLayer.predictors]
                    y_valid = self.valid_data[:][outputLayer.predictions]
                    valid_loss = outputLayer.valid(*(X_valid+y_valid))
                    # store loss
                    if best and valid_loss < best_loss:
                        best_params = np.array(lasagne.layers.get_all_param_values(self.layer_out))
                        best_loss = valid_loss
                    self.loss_record += [epoch_num+1, train_loss, valid_loss]
                    logging.info('Epoch: %d, train_loss=%f, valid_loss=%f',
                            epoch_num+1, train_loss, valid_loss)
        except KeyboardInterrupt:
            pass
        if best:
            logging.info('Reverting to best validation loss: %f', best_loss)
            lasagne.layers.set_all_param_values(self.layer_out,best_params)

    class OutputLayer():

        def __init__(self, incoming, shape, **kwargs):
            self.shape = shape
            self.name = kwargs.get('name')
            self.l_dense = lasagne.layers.DenseLayer(
                incoming,
                num_units = np.prod([s for s in shape if s]),
                name = self.name+"_dense")
            self.l_reshape = lasagne.layers.ReshapeLayer(
                self.l_dense,
                [ [0] ] + list(shape,),
                name = self.name+"_reshape")
            self.config = kwargs.get('config')
            # find inputs
            self.inputs = []
            q = [self.l_reshape] # I know it's slow shut up
            while len(q) > 0:
                l = q.pop(0)
                if hasattr(l,'input_layer'):
                    q.append(l.input_layer)
                elif hasattr(l,'input_layers'):
                    q.extend(l.input_layers)
                else:
                    self.inputs.append(l)

        @property
        def _l1_reg(self):
            return self.l_dense._l1_reg

        @_l1_reg.setter
        def _l1_reg(self,val):
            self.l_dense._l1_reg = val
            self.l_reshape._l1_reg = val

        @property
        def _l2_reg(self):
            return self.l_dense._l2_reg

        @_l2_reg.setter
        def _l2_reg(self,val):
            self.l_dense._l2_reg = val
            self.l_reshape._l2_reg = val

        def train(self, *args):
            if not hasattr(self, '_train'):
                logging.info('Compiling training function')
                shape = self.l_reshape.output_shape
                target = T.TensorType(theano.config.floatX,[False]*len(shape))()
                predictions = lasagne.layers.get_output(self.l_reshape, deterministic=False)
                loss = lasagne.objectives.squared_error(predictions, target).mean()
                updates = lasagne.updates.nesterov_momentum(
                    loss_or_grads = loss,
                    params = lasagne.layers.get_all_params(self.l_reshape),
                    learning_rate = 0.1,
                    momentum = 0.9)
                #TODO regularization weights
                l1_weights = {l:l._l1_reg for l in self.all_layers() 
                              if l._l1_reg and l.params}
                l2_weights = {l:l._l2_reg for l in self.all_layers()
                              if l._l2_reg and l.params}
                if l1_weights or l2_weights:
                    for l, w in l1_weights.items():
                        logging.info("\tlayer (%s) l1 "
                            "regularization weight: %s",l.name,w)
                    for l, w in l2_weights.items():
                        logging.info("\tlayer (%s) l2 "
                            "regularization weight: %s",l.name,w)
                else:
                    logging.info("No regularization")
                loss += regularize_layer_params_weighted(l1_weights,l1)
                loss += regularize_layer_params_weighted(l2_weights,l2)
                self._train = theano.function(
                    inputs = [l.input_var for l in self.inputs] + [target],
                    outputs = [loss],
                    updates = updates,)
            E, = self._train(*args)
            return E

        def valid(self, *args):
            if not hasattr(self, '_valid'):
                logging.info('Compiling validation function')
                shape = self.l_reshape.output_shape
                target = T.TensorType(theano.config.floatX,[False]*len(shape))()
                predictions = lasagne.layers.get_output(self.l_reshape, deterministic=True)
                loss = lasagne.objectives.squared_error(predictions, target).mean()
                self._valid = theano.function(
                    inputs = [l.input_var for l in self.inputs] + [target],
                    outputs = [loss],)
            E, = self._valid(*args)
            return E

        def __call__(self, *args, **kwargs):
            if not hasattr(self, '_call'):
                shape = self.l_reshape.output_shape
                predictions = lasagne.layers.get_output(self.l_reshape, deterministic=True)
                self._call = theano.function(
                    inputs = [l.input_var for l in self.inputs],
                    outputs = [predictions],)
            pred, = self._call(*args)
            if 'roi' in kwargs:
                pred *= roi.shape[0]
                pred += roi.offset[0]
            return pred

        def all_layers(self,cur=None,seen=None):
            if not cur: cur = self.l_reshape
            if not seen: seen = {}
            if hasattr(cur,'input_layers') and cur.input_layers:
                for inp in cur.input_layers:
                    for x in self.all_layers(inp,seen): 
                        yield x
            elif hasattr(cur,'input_layer') and cur.input_layer:
                for x in self.all_layers(cur.input_layer,seen): yield x
            if cur not in seen: yield cur

        def test(self, test_data, other, gold, loss=lasagne.objectives.squared_error, inf=10000):
            this_args = [test_data[l.name] for l in self.inputs]
            that_args = [test_data[l.name] for l in other.inputs]
            this =  self(*this_args)
            that = other(*that_args)
            this_loss = loss(this, gold).mean(axis=1)
            that_loss = loss(that, gold).mean(axis=1)
            obs_d = np.abs(this_loss.mean() - that_loss.mean())
            pooled_losses = np.append(this_loss,that_loss)
            ds = np.zeros(inf)
            logging.info('Testing')
            N = len(this_loss)
            for i in range(inf):
                np.random.shuffle(pooled_losses)
                this_pseudoloss = pooled_losses[:N]
                that_pseudoloss = pooled_losses[N:]
                ds[i] = np.abs(this_pseudoloss.mean() - that_pseudoloss.mean())
            p = np.count_nonzero(ds > obs_d) / inf
            logging.info('Model is %s %s than alternative (%f vs %f, p=%f)',
                'significantly' if p<.05 else 'non-significantly',
                'better' if this_loss.mean() < that_loss.mean() else 'worse',
                this_loss.mean(), that_loss.mean(), p)
            return (this_loss.mean(), that_loss.mean(), p)

class _FakeOutputLayer(Autotracer.OutputLayer):

    def __init__(self,X_train):
        self.guess = X_train.mean(axis=0)
        self.inputs = []

    def __call__(self,*args):
        return self.guess
