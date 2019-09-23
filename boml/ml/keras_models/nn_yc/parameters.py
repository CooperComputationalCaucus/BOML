"""
@author pmm, yu
"""

import warnings


def _default_hyperparameters():
    hyperparams = {
        # Fixed training parameters
        'shuffle': True,
        'verbose': False,
        'use_generator': False,
        'val_split': 0.2,
        'early_stopping': True,
        'patience': 5,
        'out_dir': '../test_data/test_yc',
        'run_name': 'test',
        'epochs': 10,
        # Variable training parameters (assuming use of Adam optimizer)
        'batch_size': 64,
        'lr': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'decay': 0.0,

        # Fixed model parameters
        'regression': True,
        'classification': False,
        'dense_batchnorm': True,
        'Feature_scaling': False,
        'data_shape': (10,),
        'data_fmt': 'npy',
        'dataset_dir': '../test_data/test_yc/',
        'regression_path': '../test_data/test_yc/molecule_dataset.csv',
        'regression_target': 'HER',
        'target_normalization': False,

        # Variable model parameters
        'dense_dims': [200, 100, 50],
        'dense_dropout': 0.5
    }
    return hyperparams


def load_metaparameters(param_dict=None):
    """
    Parameters for bayesian optimizer.
    Default dictionary listed and updated by param_dict.

    The default contains some redundancy dependent on the architecture.
    If the generic architecture is used, the 'dense_X' values are ignored.
    If not, the values of 'dense_units' should match the number of layers
    in the architecture and the above will be used.
    """
    metaparams = {'architecture': 'nn_yc',
                  'dense_units': 3,
                  'dense_0': 200,
                  'dense_1': 100,
                  'dense_2': 50,
                  'nodes': 100,
                  'nodes_roc': -1.,
                  'dense_dropout': 0.0,
                  'log_lr': -3,
                  'log_beta_1': -1,
                  'log_beta_2': -3
                  }
    if param_dict:
        metaparams.update(param_dict)
    metaparams['dense_units'] = int(metaparams['dense_units'])
    return metaparams


def gen_hyperparameters(metaparams):
    """
    Parameters for model architecture and training.
    Default dictionary listed and updated by metaparameters.

    First metaparameters required are used to assemble lists describing the architecture.
    Then metaparameters not required are used to update the dictionary below.
    """
    # Create list of nodes per layer from metaparams
    params = {}
    if metaparams['architecture'] == 'nn_general' or\
            metaparams['architecture'] == 'nn_yc':
        params['dense_dims'] = [metaparams['nodes']]
        for _ in range(metaparams['dense_units'] - 1):
            params['dense_dims'].append(
                int(params['dense_dims'][-1] * (2 ** metaparams['nodes_roc']))
            )
    else:
        if int(metaparams['dense_units']) != \
                int(metaparams['architecture'][-1]):
            warnings.warn('Configuration set up for {} dense units, '
                          'but architecture is {}.'.format(
                metaparams['dense_units'],
                metaparams['architecture']),
                Warning)
        params = {
            'dense_dims': [1 for _ in range(metaparams['dense_units'])]
        }
        for i in range(metaparams['dense_units']):
            key = 'dense_{}'.format(i)
            if key in metaparams:
                params['dense_dims'][i] = int(max(1, metaparams[key]))

    # Non integer params
    params['lr'] = 10 ** metaparams['log_lr']
    params['beta_1'] = 1 - 10 ** metaparams['log_beta_1']
    params['beta_2'] = 1 - 10 ** metaparams['log_beta_2']
    hyperparams = _default_hyperparameters()

    # Update single values like dropout rate, and training parameters, etc
    for key in metaparams:
        if key in hyperparams:
            hyperparams[key] = metaparams[key]
    hyperparams.update(params)
    # Enforce tuple
    hyperparams['data_shape'] = tuple(hyperparams['data_shape'])
    return hyperparams


if __name__ == '__main__':
    metaparams = load_metaparameters()
    params = gen_hyperparameters(metaparams)
    for key in params:
        print(key, params[key])
