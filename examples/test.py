#%matplotlib inline
from a3.visualize import *
from a3.utils import apil_old
import h5py
import a3
import os
import re
import logging
import argparse
from a3 import utils
#reload(logging)


logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

def train_network(config):
    if config.build_training_db:
        ds = a3.Dataset(config.training_db, roi=config.roi, n_points=config.out_points, scale=config.scale_factor)
        print("Scanning directory")
        ds.scan_directory(config.training_directory, config.training, config.keys)
        print("Reading sources...")
        ds.read_sources(['trace','image','name'])
        print(ds.sources.keys())
        print("Preparing autotracer...")
    # Create network
    a = a3.Autotracer(config.training_db, None, roi=config.roi, config=config)
    a.train(config.epochs)
    a.save(config.output_network)
    # visualize output
    Visualizer.visualize_training(a.loss_record, savefig=config.figname)

def test_network(config):
    if config.build_testing_db:
        ds = a3.Dataset(config.testing_db, roi=config.roi, n_points=config.out_points, scale=config.scale_factor)
        print("Scanning directory")
        ds.scan_directory(config.testing_directory, config.training, config.keys)
        print("Reading sources...")
        ds.read_sources(['trace','image','name'])
        print(ds.sources.keys())
        print("Preparing autotracer...")
    a = a3.Autotracer(config.training_db, config.testing_db, roi=config.roi, config=config)
    a.load(config.output_network)
    with h5py.File(config.testing_db,'r') as h:
        # trace all images used in training
        a.trace(h['image'], config.test_out, h['name'], config.tracer, config.subject_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an autotrace network with the given configuration.')
    parser.add_argument('config_file',
                        metavar='i',
                        type=str,
                        help='The path the the networkconfig.yml')

    args = parser.parse_args()
    print("Using {}".format(args.config_file))
    config = utils.Config(args.config_file)
    if config.train:
        train_network(config)
    if config.test:
        test_network(config)
