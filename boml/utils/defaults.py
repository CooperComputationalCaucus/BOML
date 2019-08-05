"""
@author: maffettone

Each new model type built should be paired with a set of default training parameters,
and variable space. The variable space is fed to the Bayesian optimization.
"""


def cnn_defaults(config):
    config['training_params'] = {'val_split': 0.2,
                                 'dataset_dir': '../test_data/organic_rgb/',
                                 'early_stopping': False,
                                 'patience': 4,
                                 'epochs': 1,
                                 'out_dir': '../test_data',
                                 'run_name': 'test',
                                 'batch_size': 8,
                                 'n_classes': 2,
                                 'data_shape': (32, 32, 3),
                                 'data_fmt': 'png',
                                 'verbose': True,
                                 'use_generator': False
                                 }
    config['variable_space'] = {'conv_units': [2, 6, 1],
                                'filters': [10, 100, 1],
                                'filters_roc': [-1.0, 1.0, 0.1],
                                'kernel_size': [2, 10, 1],
                                'kernel_roc': [-1.5, 1.0, 0.1],
                                'stride': [1, 3, 1],
                                'stride_roc': [-1.5, 1.0, 0.1],
                                'conv_dropout': [0.0, 0.1, 0.001],
                                'pool_size': [1, 4, 1],
                                'pool_roc': [-1.5, 0., .1],
                                'dense_units': [0, 2, 1],
                                'nodes': [10, 1000, 10],
                                'nodes_roc': [-2.0, 0.0, 0.1],
                                'dense_dropout': [0.0, 0.8, 0.001]
                                }
    config['fixed_space'] = {'log_lr': -3,
                             'log_beta_1': -1,
                             'log_beta_2': -3}
    return


def cnn2_defaults(config):
    return cnn_defaults(config)


def nn_defaults(config):
    return config


def rf_defaults(config):
    return config


def nb_defaults(config):
    return config


def svm_defaults(config):
    return config


def gp_defaults(config):
    return config


def default_config(architecture=None):
    config = {'verbose': 2,
              'debug_msgs': False,
              'max_iter': 3,
              'max_time': 3600,
              'init_random': 2,
              'seed': None,
              'state': 1,
              'sampler': 'greedy',
              'basename': 'test',
              'target': 'val_f1',
              'multiprocessing': 1,
              'previous_points': False,
              'feature_scaling': False,
              'regression': False,
              'classification': True,
              'variable_space': {
              },
              'fixed_space': {
              },
              'training_params': {
              }
              }

    if architecture is None:
        config['architecture'] = 'cnn'
        cnn_defaults(config)
    elif architecture == 'cnn':
        cnn_defaults(config)
    elif architecture == 'cnn2':
        cnn2_defaults(config)
    elif architecture[0:2] == 'nn':
        nn_defaults(config)
    elif architecture == 'rf':
        rf_defaults(config)
    elif architecture == 'nb':
        nb_defaults(config)
    elif architecture == 'svm':
        svm_defaults(config)
    elif architecture == 'gp':
        gp_defaults(config)
    return config
