"""
@author: maffettone
Set of checks for configurations to catch obvious issues before they arise.
"""
import warnings
#TODO: Add checks for datashapes (sklearn should be (m,))

def general_checks(config):
    if config['init_random'] == 0 & config['previous_points']:
        raise RuntimeError("A model without previous points requires random initialization.\n"
                           "Please check 'init_random' and 'previous_points' in your configuration file.")

    if config['regression'] & config['classification']:
        raise RuntimeError("Cannot simultaneously do regression and classification. Check configuration!")

def sklearn_checks(config):
    if len(config['data_shape']) != 1:
        raise RuntimeError("sklearn models require an input data shape of (n,). Please ensure your data is 1-dimensional")

    if config['training_params']['val_split'] > 0:
        warnings.WarningMessage('Validation split set to {}; however, irrelevant for sklearn models. \n'
                                'These models use cross-validation by default and split the data accordingly'.format(
            config['training_params']['val_split']))

def cnn_checks(config):
    pass


def cnn2_checks(config):
    pass


def nn_checks(config):
    pass


def rf_checks(config):
    pass


def nb_checks(config):
    pass


def svm_checks(config):
    sklearn_checks(config)

def gp_checks(config):
    pass


def sanity_checks(config):
    general_checks(config)

    if config['architecture'] == 'cnn':
        cnn_checks(config)
    elif config['architecture'] == 'cnn2':
        cnn2_checks(config)
    elif config['architecture'][0:2] == 'nn':
        nn_checks(config)
    elif config['architecture'] == 'rf':
        rf_checks(config)
    elif config['architecture'] == 'nb':
        nb_checks(config)
    elif config['architecture'] == 'svm':
        svm_checks(config)
    elif config['architecture'] == 'gp':
        gp_checks(config)
    return True
