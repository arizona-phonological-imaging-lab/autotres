#!/usr/bin/env python3

from __future__ import absolute_import, division

from .roi import ROI
from .errors import ShapeError
from .constants import _version
from lasagne.regularization import regularize_layer_params, regularize_layer_params_weighted, l2, l1
from .struct import LossRecord
from .utils import get_path
import json
import logging

import numpy as np
import h5py
import theano
import theano.tensor as T
import lasagne

class Autotracer(object):
    """Automatically traces tongues in Ultrasound images.

    Attributes:
        roi (ROI): Where the ultrasound images the data represent.
        X_train (tensor of float32): the training dataset images.
            each element is 1 pixel, scaled from 0 (black) to 1 (white).
            If roi.shape is (y,x) then X_train.shape must be (N,1,y,x).
        y_train (tensor of float32): the training dataset traces.
            Elements represent points on the tongue relative to the roi.
            0 represents that the point lies on roi.offset[0], while
            1 represents that the point lies on roi.extent[0].
            For traces with M points, y_train.shape should be (N,M,1,1).
        X_valid (tensor of float32): the validation dataset images
            each element is 1 pixel, scaled from 0 (black) to 1 (white).
            if roi.shape is (y,x) then X_valid.shape should be (N,1,y,x).
        y_valid (tensor of float32): the validation dataset traces.
            Elements represent points on the tongue relative to the roi.
            0 represents that the point lies on roi.offset[0], while
            1 represents that the point lies on roi.extent[0].
            For traces with M points, y_valid.shape should be (N,M,1,1).
        layer_in (lasagne.layers.input.InputLayer): input layer
        layer_out (lasagne.layers.dense.DesnseLayer): output layer
    """

    # TODO: alternative constructor using data from Config instance
    # TODO: incorporate network json into config
    def __init__(self, net_json, roi, train=None, valid=None, config=None):
        """

        Args:
            net_json (string): the location of a network definition file
            roi (ROI): the location of the data within an image
            train (string): the location of a hdf5 dataset for training
                this gets loaded as X_train and y_train
            valid (string): the location of a hdf5 dataset for validation
                this gets loaded as X_valid and y_valid
            config (string): the location of a network configuration file
        """
        self.__nonlinearities = {
            'relu' : lasagne.nonlinearities.rectify}
        self._layers = {}

        self.config = config
        if train:
            # clean up paths
            train = get_path(train)
            valid = get_path(valid) if valid else valid
            self.loadHDF5(train, valid)
        self.roi = ROI(roi)
        self.__init_layers(net_json)
        self.__init_model()

    @property
    def Xshape(self):
        """A dict mapping predictors to their expected shapes"""
        return {l.name:l.shape[1:] for l in self.layer_in}

    @property
    def yshape(self):
        """The shape of the output"""
        return self.layer_out.output_shape[1:]

    @property
    def predictors(shape):
        """A list of the predictors (inputs) for the net"""
        return [l.name for l in self.layer_in]

    def loadHDF5(self,train,valid=None):
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
        with h5py.File(train,'r') as h:
            self.X_train = {k:np.array(h[k]) for k in h}
            self.y_train = np.array(h['trace'])
        if valid:
            with h5py.File(valid,'r') as h:
                self.X_valid = {np.array(h[k]) for k in h}
                self.y_valid = np.array(h['trace'])
        else:
            # split the training data into a training set and a validation set.
            i = {np.floor(self.X_train[k].shape[0] * 0.75) for k in self.X_train}
            if len(i) is not 1:
                raise Exception("Different N for diferent X")
            i = i.pop()
            self.X_valid = {k:self.X_train[k][i:] for k in self.X_train}
            self.y_valid = self.y_train[i:]
            self.X_train = {k:self.X_train[k][:i] for k in self.X_train}
            self.y_train = self.y_train[:i]
        mismatch = False
        if any((self.X_valid[k].shape[1:] != self.X_train[k].shape[1:] for k in self.X_valid)):
            logging.warn("Train and valid set have different input shape")
            mismatch = True
        if self.y_valid.shape[1:] != self.y_train.shape[1:]:
            logging.warn("Train and valid set have different output shape")
            mismatch = True
        if mismatch:
            raise ShapeError({k:self.X_valid[k].shape for k in self.X_valid},
                self.y_valid.shape)

    def __init_layers(self,jfile):
        """Create the architecture of the MLP

        Args:
            jfile (string): Location of a json file specifying the 
                desired architecture for the network. 
                See examples/ for example files

        Raises:
            ShapeError: if input and/or output dimensionality are unset
        """

        from .utils import compressed_file
        with compressed_file(jfile,'rt') as f:
            d = json.load(f)
        self.layer_in = [] # will be filled by __init_layers_file_recursive
        self.__init_layers_file_recursive(d)
        self.layer_out = self._layers['trace']

        # For regularization (see: http://lasagne.readthedocs.org/en/latest/modules/regularization.html)
        config = self.config
        input_layer_weight = 0.1 if not config else config.l2_input_layer_weight
        output_layer_weight = 0.5 if not config else config.l2_output_layer_weight
        self.layer_weights = {self.layer_out: output_layer_weight}
        for layer in self.layer_in:
            self.layer_weights[layer] = input_layer_weight

    def __init_layers_file_recursive(self,d,cur='trace',encoding='IBM500'):
        """Recursively traverse the architecture definition in d

        Args:
            cur (string): The key of the current layer. d[cur] should be 
                a dict describing a layer of the network
            encoding (string): A python built-in encoding for loading 
                the binary bytestring from JSON. Defaults to EBCDIC.
        """
        if cur in self._layers:
            return self._layers[cur]
        l_type = d[cur]['type']
        if l_type == 'dense':
            l_input = self.__init_layers_file_recursive(d,d[cur]['input'])
            l_nl = (self.__nonlinearities[d[cur]['nonlinearity']] 
                if 'nonlinearity' in d[cur] 
                else lasagne.nonlinearities.rectify)
            dtype = (d[cur]['dtype'] if 'dtype' in d[cur] 
                else theano.config.floatX)
            l_num_units = int(d[cur]['num_units'])
            l_W = (np.fromstring(d[cur]['W'].encode(encoding),dtype).reshape((-1,l_num_units)) 
                if 'W' in d[cur] else lasagne.init.GlorotUniform())
            l_b = (np.fromstring(d[cur]['b'].encode(encoding),dtype) 
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
            elif hasattr(self,'X_train'):
                l_shape = self.X_train[cur].shape[1:]
            else:
                raise RuntimeError('Cannot guess shape for input "%s"' % (cur,))
            l_shape = (None,) + l_shape
            l = lasagne.layers.InputLayer(
                shape = l_shape,
                name = cur)
            if cur not in {l.name for l in self.layer_in}:
                self.layer_in.append(l)
        elif l_type == 'dropout':
            l_input = self.__init_layers_file_recursive(d,d[cur]['input'])
            l_p = float(d[cur]['p']) if 'p' in d[cur] else 0.5
            l = lasagne.layers.DropoutLayer(
                l_input,
                p = l_p,
                name = cur)
        elif l_type == 'concat':
            l_inputs = [self.__init_layers_file_recursive(d,k)
                for k in d[cur]['inputs']]
            l_axis = d[cur]['axis'] if 'axis' in d[cur] else 1
            l = lasagne.layers.ConcatLayer(
                l_inputs,
                axis = l_axis,
                name = cur)
        else:
            raise NotImplementedError("Cannot (yet) load %s layers."%(l_type))
        self._layers[cur] = l
        return l

    def save(self,fname,save_params=None,compress=None,encoding='IBM500'):
        """Save the current network to a file
 
        Args:
            fname (string): Where to save the network.
            save_params (bool): Whether or not to save netowrk weights
                A value of None (default) will save only if it will be 
                compressed.
            compress (file construtor): How to compress the file. Should
                return a file-like object; e.g. gzip.open. False results
                in an uncompressed file, while None (default) makes an
                intelligent decision based on file name.
            encoding (string): A python built-in encoding for saving
                the binary bytestring as JSON. Defaults to EBCDIC.
        """
        if compress == None:
            from .utils import compressed_file
            compress = compressed_file
        elif compress == True:
            import bz2
            compress = bz2.open
            fname += '.bz2'
        elif not compress:
            compress = open
        d = {}
        self.__save_recursive(d,self.layer_out,save_params,encoding)
        with compress(fname,'wt') as f:
            if save_params == None: 
                import io
                save_params = (type(f) != io.TextIOWrapper)
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
        elif type(layer) == lasagne.layers.input.InputLayer:
            t['type'] = 'input'
            t['shape'] = list(layer.shape[1:])
        d[i] = t
        return i

    def __init_model(self):
        """Initializes the model and compiles the network

        For the most part, this consists of setting up some bookkeeping
        for theano and lasagne, and compiling the theano functions
        """
        logging.info('initializing model')
        if self.Xshape == None or self.yshape == None:
            if self.Xshape == None:
                logging.warning("Tried to compile Neural Net before"
                    "setting input dimensionality")
            if self.yshape == None:
                logging.warning("Tried to compile Neural Net before"
                    "setting output dimensionality")
            raise ShapeError(self.Xshape,self.yshape)

        # These are theano/lasagne symbolic variable declarationss,
        # representing... the target vector(traces)
        target_vector = T.fmatrix('y')
        # our predictions
        predictions = lasagne.layers.get_output(self.layer_out)
        validation_predictions = lasagne.layers.get_output(self.layer_out, deterministic=True)
        # the loss (diff in objective) for training
        # using MSE
        stochastic_loss = lasagne.objectives.squared_error(predictions, target_vector).mean()
        #print(stochastic_loss)
        deterministic_loss = lasagne.objectives.squared_error(validation_predictions, target_vector).mean()
        # using cross entropy
        #stochastic_loss = lasagne.objectives.categorical_crossentropy(predictions, target_vector).mean()
        # the loss for validation
        #deterministic_loss = lasagne.objectives.categorical_crossentropy(test_predictions, target_vector).mean()
        # calculate loss
        loss = stochastic_loss
        # should regularization be used?
        config = self.config
        if config:
            if config.l1_regularization:
                logging.info("Using L1 regularization")
                l1_penalty = regularize_layer_params(self.layer_out, l1) * 1e-4
                loss += l1_penalty
            if config.l2_regularization:
                logging.info("Using L2 regularization with weights")
                for sublayer in self.layer_in:
                    logging.info("\tinput layer ({1}) weight: {0}".format(self.layer_weights[sublayer],sublayer.name))
                logging.info("\toutput layer weight: {0}".format(self.layer_weights[self.layer_out]))
                l2_penalty = regularize_layer_params_weighted(self.layer_weights, l2)
                loss += l2_penalty
        else:
            logging.info("No regularization")
        # the network parameters (i.e. weights)
        all_params = lasagne.layers.get_all_params(
            self.layer_out)
        # how to update the weights
        updates = lasagne.updates.nesterov_momentum(
            loss_or_grads = loss,
            params = all_params,
            learning_rate = 0.1,
            momentum = 0.9)

        # The theano functions for training, validating, and tracing.
        #   These get method-level wrappers below
        logging.info('compiling theano functions')
        self._train_fn = theano.function(
            on_unused_input='warn',
            inputs  = [l.input_var for l in self.layer_in]+[target_vector],
            outputs = [stochastic_loss],
            updates = updates)
        self._valid_fn = theano.function(
            on_unused_input='warn',
            inputs  = [l.input_var for l in self.layer_in]+[target_vector],
            outputs = [deterministic_loss,
                lasagne.layers.get_output(self.layer_out)])
        self._trace_fn = theano.function(
            on_unused_input='warn',
            inputs  = [l.input_var for l in self.layer_in],
            outputs = [lasagne.layers.get_output(self.layer_out)
                * self.roi.shape[0] + self.roi.offset[0]])



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

    def train(self,num_epochs=2500,batch_size=512):
        """Train the MLP using minibatches

        Args:
            num_epochs (int): Number of times to run through the
                training set during each epoch.
            batch_size (int): Number of images to calculate updates on
        """
        logging.info('Training')
        if not all((hasattr(self,x) for x in 
                   ('X_train','X_valid','y_train','y_valid'))):
            logging.warning('Cannot train without training data!')
            return False
        # keep track of (epoch + 1, train_loss, valid_loss)
        self.loss_record = LossRecord()
        for epoch_num in range(num_epochs):
            num_batches_train = int(np.ceil(len(self.X_train) / batch_size))
            train_losses = []
            for batch_num in range(num_batches_train):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num +1))
                X_batch = [self.X_train[l.name][batch_slice] for l in self.layer_in]
                y_batch = self.y_train[batch_slice,:,0,0]
                loss, = self.train_batch(*(X_batch+[y_batch]))
                train_losses.append(loss)
            train_loss = np.mean(train_losses)
            num_batches_valid = int(np.ceil(len(self.X_valid) / batch_size))
            valid_losses = []
            list_of_traces_batch = []
            for batch_num in range(num_batches_valid):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num + 1))
                X_batch = [self.X_valid[l.name][batch_slice] for l in self.layer_in]
                y_batch = self.y_valid[batch_slice,:,0,0]
                loss, traces_batch = self.valid_batch(*(X_batch+[y_batch]))
                valid_losses.append(loss)
                list_of_traces_batch.append(traces_batch)
            valid_loss = np.mean(valid_losses)
            # store loss
            self.loss_record += [epoch_num+1, train_loss, valid_loss]
            logging.info('Epoch: %d, train_loss=%f, valid_loss=%f'
                    % (epoch_num+1, train_loss, valid_loss))
