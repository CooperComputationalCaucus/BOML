"""
@author: maffettone

Parameters set up for support vector machines.
"""


def _default_hyperparameters():
    hyperparams = {
        # Fixed training parameters
        'shuffle': True,
        'verbose': False,
        'val_split': 0.0,
        'out_dir': '../test_data',
        'run_name': 'test',
        'cv': 5,
        'n_jobs': 0,
        'cv_threads': 1,
        'architecture': 'svm',

        # Variable training parameters for each model type
        'gamma': 2.0,
        'C': 1.0,

        # Fixed model parameters
        'n_classes': 16,
        'data_shape': (3072,),
        'data_fmt': 'png',
        'dataset_dir': '../test_data/organic_rgb/',
        'regression_path': None,
        'regression_target': None,
        'target_normalization': False
    }
    return hyperparams


def load_metaparameters(param_dict=None):
    """
    Parameters for bayesian optimizer. Default dictionary listed and updated by param_dict.
    """
    metaparams = {'architecture': 'svm',
                  'log_gamma': -3,
                  'log_C': -2}
    if param_dict:
        metaparams.update(param_dict)

    return metaparams


def gen_hyperparameters(metaparams):
    """
    Parameters for model architecture and training. Default dictionary listed and updated by metaparameters.

    First metaparameters required are used to assemble lists describing the architecture.
    Then metaparameters not required are used to update the dictionary below.
    """

    hyperparams = _default_hyperparameters()
    hyperparams['C'] = 10 ** metaparams['log_C']
    hyperparams['gamma'] = 10 ** metaparams['log_gamma']

    # Update single values like cross val, directories, etc
    for key in metaparams:
        if key in hyperparams:
            hyperparams[key] = metaparams[key]

    # Ensure types
    hyperparams['cv'] = int( hyperparams['cv'])

    # Multiprocessing for sklearn
    if metaparams['multiprocessing']:
        hyperparams['n_jobs'] = min(hyperparams['cv'],metaparams['multiprocessing'])
    else:
        hyperparams['n_jobs'] = 1
    return hyperparams


if __name__ == '__main__':
    metaparams = load_metaparameters()
    params = gen_hyperparameters(metaparams)
    for key in params:
        print(key, params[key])
