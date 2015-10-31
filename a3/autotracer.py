#!/usr/bin/env pytohn3

from __future__ import absolute_import, division

from .roi import ROI
from .errors import ShapeError
from .constants import _version

import json
import logging

# These evironment flags set up GPU training
import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu'

import numpy as np
import h5py
import theano
import theano.tensor as T
import lasagne

class Autotracer(object):
    """Automatically traces tongues in Ultracound images.
    
    Attributes (all read-only):
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
            if roi.shape is (y,x) then X_test.shape should be (N,1,y,x).
        y_valid (tensor of float32): the validation dataset traces.
            Elements represent points on the tongue relative to the roi.
            0 represents that the point lies on roi.offset[0], while
            1 represents that the point lies on roi.extent[0].
            For traces with M points, y_valid.shape should be (N,M,1,1).
        layer_in (lasagne.layers.input.InputLayer): input layer
        layer_out (lasagne.layers.dense.DesnseLayer): output layer
    """
    def __init__(self,train,test,roi):
        """
        Currently, saving and loading autotracers is not supported, 
        so only one constructor syntax is supported. 

        Args:
            train (string): the location of a hdf5 dataset for training
                this gets loaded as X_train and y_train
            test (string): the location of a hdf5 dataset for validation
                this gets loaded as X_valid and y_valid
            roi (ROI): the location of the data within an image
        """
        self.loadHDF5(train,test)
        self.roi = ROI(roi)
        self.__init_model()

    def loadHDF5(self,train,test=None):
        """Load a test and training dataset from hdf5 databases

        Args:
            train (string): the location of a hdf5 dataset for training
                this gets loaded as X_train and y_train
            test (string): the location of a hdf5 dataset for validation
                this gets loaded as X_valid and y_valid

        Raises:
            ShapeError: if train and test have incompatible shape
                Xshape and yshape are set to the shape of test
        """
        logging.debug('loadHDF5(%s,%s)' % (train,test))
        with h5py.File(train,'r') as h:
            self.X_train = np.array(h['image'])
            self.y_train = np.array(h['trace'])
            self.Xshape = self.X_train.shape[1:]
            self.yshape = self.y_train.shape[1:]
        if test:
            with h5py.File(test,'r') as h:
                self.X_valid = np.array(h['image'])
                self.y_valid = np.array(h['trace'])
        else: 
            # split the training data into a training set and a validation set. 
            i = np.floor(self.X_train.shape[0] * 0.75)
            self.X_valid = self.X_train[i:]
            self.y_valid = self.y_train[i:]
            self.X_train = self.X_train[:i]
            self.y_train = self.y_train[:i]
        mismatch = False
        if self.X_valid.shape[1:] != self.Xshape:
            logging.warn("Train and test set have different input shape")
            mismatch = True
        if self.y_valid.shape[1:] != self.yshape:
            logging.warn("Train and test set have different output shape")
            mismatch = True
        if mismatch:
            raise ShapeError(self.X_valid.shape[1:],self.y_valid.shape[1:])

    def __init_layers(self,layer_size):
        """Create the architecture of the MLP
        
        The achitecture is currently hard-coded.
        Architecture:
            image -> ReLU w/ dropout (x3) -> trace
        Args:
            layer_size (integer): the size of each layer
                Currently, all the layers have the same number of units
                (except for the input and output layers).

        Raises:
            ShapeError: if input and/or output dimensionality are unset
        """
        if self.Xshape == None or self.yshape == None:
            if self.Xshape == None:
                logging.warning("Tried to compile Neural Net before" +
                    "setting input dimensionality")
            if self.yshape == None:
                logging.warning("Tried to compile Neural Net before" +
                    "setting output dimensionality")
            raise ShapeError(self.Xshpae,self.yshape)
        self.layer_in = lasagne.layers.InputLayer(
            shape = (None,) + self.Xshape)
        l_hidden1 = lasagne.layers.DenseLayer(
            self.layer_in,
            num_units = layer_size,
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())
        l_hidden1_d = lasagne.layers.DropoutLayer(l_hidden1, p=.5)

        l_hidden2a = lasagne.layers.DenseLayer(
            l_hidden1_d,
            num_units = layer_size,
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())
        l_hidden2a_d = lasagne.layers.DropoutLayer(l_hidden2a, p=.5)
        l_hidden3a = lasagne.layers.DenseLayer(
            l_hidden2a_d,
            num_units = self.yshape[0],
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())

        l_hidden2b = lasagne.layers.DenseLayer(
            l_hidden1_d,
            num_units = layer_size,
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())
        l_hidden2b_d = lasagne.layers.DropoutLayer(l_hidden2b, p=.5)
        l_hidden3b = lasagne.layers.DenseLayer(
            l_hidden2b_d,
            num_units = self.yshape[0],
            nonlinearity = theano.tensor.nnet.hard_sigmoid,
            W = lasagne.init.GlorotUniform())

        self.layer_out = lasagne.layers.ElemwiseMergeLayer(
            [l_hidden3a,l_hidden3b],
            merge_function=theano.tensor.mul)

    def __init_model(self,layer_size=2048):
        """Initializes the model
        
        For the most part, this consists of setting up some bookkeeping
        for theano and lasagne, and compiling the theano functions
        Args:
            layer_size (integer): the size of each layer in the MLP
            gets passed directly to self.__init_layers
        """
        logging.info('initializing model')
        self.__init_layers(layer_size)

        # These are theano/lasagne symbolic variable declarationss, 
        # representing... the target vector(traces)
        target_vector = T.fmatrix('y')
        # the loss (diff in objective) for training
        stochastic_loss = lasagne.objectives.squared_error( 
            lasagne.layers.get_output(self.layer_out),target_vector).mean()
        # the loss for validation
        deterministic_loss = lasagne.objectives.squared_error( 
            lasagne.layers.get_output(self.layer_out,
            deterministic=True),target_vector).mean()
        # the network parameters (i.e. weights)
        all_params = lasagne.layers.get_all_params( 
            self.layer_out)
        # how to update the weights
        updates = lasagne.updates.nesterov_momentum( 
            loss_or_grads = stochastic_loss, 
            params = all_params,
            learning_rate = 0.1,
            momentum = 0.9)
        
        # The theano functions for training, testing, and tracing. 
        #   These get method-level wrappers below
        logging.info('compiling theano functions')
        self._train_fn = theano.function(
            inputs  = [self.layer_in.input_var,target_vector],
            outputs = [stochastic_loss],
            updates = updates)
        self._valid_fn = theano.function(
            inputs  = [self.layer_in.input_var, target_vector],
            outputs = [deterministic_loss,
                lasagne.layers.get_output(self.layer_out)])
        self._trace_fn = theano.function(
            inputs  = [self.layer_in.input_var],
            outputs = [lasagne.layers.get_output(self.layer_out)
                * self.roi.shape[0] + self.roi.offset[0]])
    
    def train_batch(self, X, y):
        """Train on a minibatch 
        
        Wrapper for _train_fn()
        
        Args:
            X (tensor of float32): Minibatch from the training images
            y (tensor of float32): The corresponding traces
        """
        return self._train_fn(X,y)

    def valid_batch(self, X, y):
        """Validates the network on a (mini)batch

        Wrapper for _valid_fn()

        Args:
            X (tensor of float32): Minibatch from the validation images
            y (tensor of float32): The corresponding traces
        """
        return self._valid_fn(X,y)

    def trace(self, X, jfile=None,names=None,project_id=None,subject_id=None):
        """Trace a batch of images using the MLP

        Can be used programmatically to get a numpy array of traces, 
        or a json file for use with the APIL webapp.
        Args:
            X (tensor of float32): image to be traced
                should be properly scaled to [0,1] and the roi.
            jfile (string, optional): location to save json traces
                If falsey, then no json trace is created
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
        t, = self._trace_fn(X)
        if jfile:
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

    def save(self,fname):
        params = np.array(lasagne.layers.get_all_param_values(self.layer_out))
        np.save(fname,params)

    def load(self,fname):
        params = np.load(fname)
        lasagne.layers.set_all_param_values(self.layer_out,params)

    def train(self,num_epochs=2500,batch_size=512):
        """Train the MLP using minibatches

        Args:
            num_epochs (int): Number of times to run through the 
                training set during each epoch.
            batch_size (int): Number of images to calculate updates on
        """
        logging.info('Training')
        for epoch_num in range(num_epochs):
            num_batches_train = int(np.ceil(len(self.X_train) / batch_size))
            train_losses = []
            for batch_num in range(num_batches_train):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num +1))
                X_batch = self.X_train[batch_slice]
                y_batch = self.y_train[batch_slice,:,0,0]
                loss, = self.train_batch(X_batch, y_batch)
                train_losses.append(loss)
            train_loss = np.mean(train_losses)
            num_batches_valid = int(np.ceil(len(self.X_valid) / batch_size))
            valid_losses = []
            list_of_traces_batch = []
            for batch_num in range(num_batches_valid):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num + 1))
                X_batch = self.X_valid[batch_slice]
                y_batch = self.y_valid[batch_slice,:,0,0]
                loss, traces_batch = self.valid_batch(X_batch, y_batch)
                valid_losses.append(loss)
                list_of_traces_batch.append(traces_batch)
            valid_loss = np.mean(valid_losses)
            logging.info('Epoch: %d, train_loss=%f, valid_loss=%f'
                    % (epoch_num+1, train_loss, valid_loss))    

