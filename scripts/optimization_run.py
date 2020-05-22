"""
Script for running general optimization from configuration file

@author: maffettone
"""
import sys
sys.path.append('../')
from boml.optimization import optimize_nets
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help='config file (.json)')
    args = vars(parser.parse_args())
    if args['config']:
        optimize_nets(config_file=args['config'])
    else:
        optimize_nets()