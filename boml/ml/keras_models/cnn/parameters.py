"""
@author: pmm
Configuration parameters are global and for the optimizer.
Metaparameters are used to generate hyperparameters (for instance by making logarithmic searching easy).
Hyperparameters are passed to a given training run.
As such, there are many shared key:value pairs between the metaparameter and hyperparameter dictionaries.
"""


def _default_hyperparameters():
    hyperparams = {
        # Fixed training parameters
        'shuffle': True,
        'verbose': False,
        'use_generator': False,
        'val_split': 0.2,
        'early_stopping': True,
        'patience': 5,
        'out_dir': '../test_data',
        'run_name': 'test',
        'epochs': 10,
        # Variable training parameters (assuming use of Adam optimizer)
        'batch_size': 64,
        'lr': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'decay': 0.0,

        # Fixed model parameters
        'conv_batchnorm': True,
        'dense_batchnorm': True,
        'n_classes': 16,
        'data_shape': (32, 32, 3),
        'data_fmt': 'png',
        'dataset_dir': '../test_data/organic_rgb/',
        # Variable model parameters
        'conv_units': 3,  # redundant, but useful
        'conv_filters': [80, 80, 80],
        'conv_kernels': [100, 50, 25],
        'conv_strides': [5, 5, 2],
        'conv_dropout': 0.3,
        'pool_sizes': [3, 3, 3],
        'dense_units': 2,
        'dense_dims': [200, 100],
        'dense_dropout': 0.5
    }
    return hyperparams


def load_metaparameters(param_dict=None):
    """
    Parameters for bayesian optimizer. Default dictionary listed and updated by param_dict.
    """
    metaparams = {
        'conv_units': 3,
        'filters': 80,
        'filters_roc': 0.,
        'kernel_size': 100,
        'kernel_roc': -1.,
        'stride': 5,
        'stride_roc': -0.5,
        'conv_dropout': 0.3,
        'pool_size': 3,
        'pool_roc': 0.,
        'dense_units': 2,
        'nodes': 2300,
        'nodes_roc': -1.,
        'dense_dropout': 0.5,
    }
    if param_dict: metaparams.update(param_dict)
    metaparams['conv_units'] = int(metaparams['conv_units'])
    metaparams['dense_units'] = int(metaparams['dense_units'])
    return metaparams


def gen_hyperparameters(metaparams):
    """
    Parameters for model architecture and training. Default dictionary listed and updated by metaparameters.

    First metaparameters required are used to assemble lists describing the architecture.
    Then metaparameters not required are used to update the dictionary below.
    """

    # Create lists from metaparams
    params = {
        'conv_filters': [metaparams['filters']],
        'conv_kernels': [metaparams['kernel_size']],
        'conv_strides': [metaparams['stride']],
        'pool_sizes': [metaparams['pool_size']],
        'dense_dims': [],
    }
    if metaparams['dense_units'] > 0: params['dense_dims'] = [metaparams['nodes']]
    for _ in range(metaparams['dense_units'] - 1):
        params['dense_dims'].append(params['dense_dims'][-1] * (2 ** metaparams['nodes_roc']))
    for _ in range(metaparams['conv_units'] - 1):
        params['conv_filters'].append(params['conv_filters'][-1] * (2 ** metaparams['filters_roc']))
        params['conv_kernels'].append(params['conv_kernels'][-1] * (2 ** metaparams['kernel_roc']))
        params['conv_strides'].append(params['conv_strides'][-1] * (2 ** metaparams['stride_roc']))
        params['pool_sizes'].append(params['pool_sizes'][-1] * (2 ** metaparams['pool_roc']))
    # Ensure validity
    valid = lambda x: max(1, round(x))
    for key in params:
        params[key] = list(map(int, (map(valid, params[key]))))

    hyperparams = _default_hyperparameters()

    # Update single values like dropout rate, learning rate, etc
    for key in metaparams:
        if key in hyperparams: hyperparams[key] = metaparams[key]
    hyperparams.update(params)
    # Enforce tuple
    hyperparams['data_shape'] = tuple(hyperparams['data_shape'])
    return hyperparams


def load_hyperparameters(params_file=None, params_dict=None):
    """
    load hyperparameters for a single run from a json or a dictionary
    """
    import json
    if params_file:
        with open(params_file, 'r') as f:
            params_file = json.load(f)

    params = _default_hyperparameters()
    if params_file: params.update(params_file)
    if params_dict: params.update(params_dict)
    return params


if __name__ == '__main__':
    metaparams = load_metaparameters()
    params = gen_hyperparameters(metaparams)
    for key in params:
        print(key, params[key])
