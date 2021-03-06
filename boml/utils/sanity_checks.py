"""
@author: maffettone
Set of checks for configurations to catch obvious issues before they arise.
"""
import warnings
import os


def custom_formatwarning(message, category, filename, lineno, *args, **kwargs):
    return "{}:{}: {}: {} \n".format(filename, lineno, category, message)


warnings.formatwarning = custom_formatwarning


def general_checks(config):
    if config['init_random'] == 0 and (not config['previous_points']):
        raise RuntimeError("A model without previous points requires random initialization.\n"
                           "Please check 'init_random' and 'previous_points' in your configuration file.")

    if config['regression']:
        if config['classification']:
            raise RuntimeError("Cannot simultaneously do regression and classification. Check configuration!")
        if 'regression_path' not in config['training_params'] or not os.path.exists(config['training_params']['regression_path']):
            raise RuntimeError("No regression path or nonexistent regression path given.")


def sklearn_checks(config):
    if len(config['training_params']['data_shape']) != 1:
        raise RuntimeError(
            "sklearn models require an input data shape of (n,). Please ensure your data is 1-dimensional")

    if 'val_split' in config['training_params']:
        if 'cv' not in config['training_params'] or config['training_params']['cv']<1:
            try:
                config['training_params']['cv'] = int(1/config['training_params']['val_split'])
            except ZeroDivisionError:
                config['cv'] = 1
            warnings.warn(
                'Validation split set to {}; however, irrelevant for sklearn models. These models use cross-validation by '
                'default and split the data accordingly.\n'
                'Cross validation being set to {} splits'.format(
                    config['training_params']['val_split'], config['training_params']['cv'])
            )
        else:
            warnings.warn(
                'Validation split set to {}; however, irrelevant for sklearn models. These models use cross-validation by '
                'default and split the data accordingly.\n'
                'Cross validation being set to {} splits'.format(
                    config['training_params']['val_split'], config['training_params']['cv'])
            )




def cnn_checks(config):
    pass


def cnn2_checks(config):
    pass


def nn_checks(config):
    pass


def rf_checks(config):
    sklearn_checks(config)


def nb_checks(config):
    pass


def svm_checks(config):
    sklearn_checks(config)


def gp_checks(config):
    sklearn_checks(config)


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
