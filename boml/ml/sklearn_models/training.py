from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_validate
from numpy.random import RandomState

import warnings
import numpy as np
import time
import sys
import traceback
import gc

from boml.ml.utils.mem_utils import check_dataset_memory
from boml.ml.utils.data_proc import ListAddressesLabels, load_categorical_data, load_regression_data


### REQUIRES CHANGES FOR NEW MODEL TYPES ###
def gen_model(params):
    if params['architecture'] == 'svm':
        from .svm.models import gen_model as gm
    elif params['architecture'] == 'rf':
        from .rf.models import gen_model as gm
    elif params['architecture'] == 'gb':
        from .gb.models import gen_model as gm
    elif params['architecture'] == 'gp':
        from .gp.models import gen_model as gm
    else:
        raise KeyError("Invalid architecture encountered in training.py")
    return gm(params)
### REQUIRES CHANGES FOR NEW MODEL TYPES ###

def metrics_dict(params):
    if params['classification']:
        d = {'f1': make_scorer(f1_score, average='macro'),
             'acc': make_scorer(accuracy_score)
             }
    elif params['regression']:
        d = {'mse': make_scorer(mean_squared_error),
             'r2': make_scorer(r2_score)
             }
    else:
        raise ValueError("Both regression and classification are set True.")
    return d


def general_training(params):
    start_time = time.time()

    if 'seed' in params and params['seed']:
        np.random.seed(params['seed'])

    estimator = gen_model(params)
    addrs = ListAddressesLabels(params)
    train_shape = (len(addrs['train_addrs']), *params['data_shape'])
    assert train_shape[0] > 0, "No examples in training set"
    check_dataset_memory(len(addrs['train_addrs']) + len(addrs['val_addrs']),
                         params['data_shape'],
                         batch_size=train_shape[0])

    if params['classification']:
        gen_params = {'dim': params['data_shape'],
                      'shuffle': params['shuffle'],
                      'data_fmt': params['data_fmt'],
                      'one_hot': False}
        x, y = load_categorical_data(list_IDs=addrs['train_addrs'],
                                     labels=addrs['train_labels'],
                                     **gen_params)
    else:  # regression
        gen_params = {'dim': params['data_shape'],
                      'shuffle': params['shuffle'],
                      'data_fmt': params['data_fmt'],
                      'regression_target': params['regression_target'],
                      'target_normalization': params['target_normalization']
                      }
        x, y = load_regression_data(list_IDs=addrs['train_addrs'],
                                    dataset_path=params['regression_path'],
                                    **gen_params)
    scoring = metrics_dict(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(estimator, x, y,
                                scoring=scoring,
                                cv=params['cv'],
                                n_jobs=params['cv_threads'],
                                return_train_score=True)

    training_results = {}
    for key in scores:
        if key[0:5] == 'test_':
            training_results['val_{}'.format(key[5:])] = np.mean(scores[key])
    training_results['default'] = 0.
    # Negating values to minimize
    if 'val_loss' in training_results:
        training_results['val_loss'] *= -1
    if 'val_mse' in training_results:
        training_results['val_mse'] *= -1

    if params['verbose']:
        print("Time to complete run {}: {}".format(params['run_name'],
                                                   time.time() - start_time))
        print("Results:")
        for key in training_results:
            print("{} : {}".format(key, training_results[key]))

    # Forced deletions
    del x
    del y
    del scores
    del estimator
    gc.collect()

    return training_results

def training(params):
    default_results = {'val_loss': 0.,
                       'val_acc': 0.,
                       'val_f1': 0.,
                       'val_recall': 0.,
                       'val_precision': 0.,
                       'val_r2': 0.,
                       'val_mse': 0.}
    try:
        return general_training(params)
    except ValueError as err:
        if params['verbose']:
            print(err.args)
        return default_results
    except Exception as err:
        if params['verbose']:
            print("Unexpected error: ", sys.exc_info()[0])
            traceback.print_tb(err.__traceback__)
        return default_results
