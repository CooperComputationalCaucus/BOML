"""
@author: maffettone

Parameters setup for gradient boosting.
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
        'n_jobs': 1,
        'cv_threads': 1,
        'architecture': 'gb',

        # Variable training parameters for each model type
        'learning_rate': 0.1,  # learning rate shrinks the contribution of each tree by learning_rate
        'n_estimators': 10,  # The number of trees in the forest.
        'min_samples_split': 2,  # The minimum number of samples required to be at a leaf node.
        'max_features': 0,  # The fractional number of features to consider when looking for the best split. 0 -> sqrt

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
    metaparams = {'architecture': 'gb',
                  'multiprocessing': 0,
                  'log_learning_rate': -1.,
                  'log_n_estimators': 1,
                  'min_samples_split': 2,
                  'max_features': 0}
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
    hyperparams['learning_rate'] = 10 ** metaparams['log_learning_rate']
    hyperparams['n_estimators'] = 10 ** metaparams['log_n_estimators']

    # Update single values like cross val, directories, etc
    for key in metaparams:
        if key in hyperparams:
            hyperparams[key] = metaparams[key]

    # Ensure types
    hyperparams['cv'] = int(hyperparams['cv'])
    hyperparams['n_estimators'] = int(hyperparams['n_estimators'])
    hyperparams['min_samples_split'] = int(hyperparams['min_samples_split'])
    if hyperparams['max_features'] <= 0:
        hyperparams['max_features'] = 'sqrt'

    # Multiprocessing for sklearn
    if metaparams['multiprocessing']:
        hyperparams['n_jobs'] = min(hyperparams['cv'], metaparams['multiprocessing'])
    else:
        hyperparams['n_jobs'] = 1
    return hyperparams


if __name__ == '__main__':
    metaparams = load_metaparameters()
    params = gen_hyperparameters(metaparams)
    for key in params:
        print(key, params[key])
