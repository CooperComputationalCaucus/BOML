"""
@author: maffettone

Training module for all keras based models. Models and parameters should be placed in
local packages
"""
from keras import optimizers
from keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed

import numpy as np
import time
import os
import sys
import pickle
import traceback

from boml.ml.utils.mem_utils import reset_keras, config_keras, check_dataset_memory
from boml.ml.utils.data_proc import ListAddressesLabels, CategoricalDataGenerator, load_categorical_data, \
    load_regression_data
from boml.ml.utils.callbacks import ClassificationMetrics, RegressionMetrics, GeneratorClassificationMetrics, \
    GeneratorRegressionMetrics

### REQUIRES CHANGES FOR NEW MODEL TYPES ###
def gen_model(params):
    if params['architecture'] == 'cnn':
        from .cnn.models import gen_model as gm
    elif params['architecture'] == 'cnn2':
        from .cnn2.models import gen_model as gm
    elif params['architecture'][:2] == 'nn':
        from .nn.models import gen_model as gm
    else:
        from .cnn.models import gen_model as gm
    return gm(params)
### REQUIRES CHANGES FOR NEW MODEL TYPES ###

def general_training(params):
    # FIXME: Stalls in model.fit_generator() with multiprocessing
    # TODO: Regression generator implementation
    start_time = time.time()
    config_keras()

    if 'seed' in params and params['seed']:
        np.random.seed(params['seed'])
        set_random_seed(params['seed'])

    model = gen_model(params)
    addrs = ListAddressesLabels(params)
    train_shape = (len(addrs['train_addrs']), *params['data_shape'])
    val_shape = (len(addrs['val_addrs']), *params['data_shape'])
    assert train_shape[0] > 0, "No examples in training set"
    assert val_shape[0] > 0, "No examples in validation set"
    check_dataset_memory(len(addrs['train_addrs']) + len(addrs['val_addrs']),
                         params['data_shape'],
                         params['batch_size'],
                         params['use_generator'])

    # Splitting generator and complete dataset loading with some different variable conventions
    # Initialize to catch errors
    train_generator = None
    val_generator = None
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    steps_per_epoch = 0
    val_steps = 0
    if params['use_generator']:
        gen_params = {'dim': params['data_shape'],
                      'batch_size': params['batch_size'],
                      'shuffle': params['shuffle'],
                      'data_fmt': params['data_fmt'],
                      'num_classes': np.shape(np.unique(addrs['train_labels']))[0]}
        train_generator = CategoricalDataGenerator(list_IDs=addrs['train_addrs'],
                                                   labels=addrs['train_labels'],
                                                   **gen_params)
        val_generator = CategoricalDataGenerator(list_IDs=addrs['val_addrs'],
                                                 labels=addrs['val_labels'],
                                                 **gen_params)
        steps_per_epoch = max(1,len(train_generator))
        val_steps = max(1,len(val_generator))
    else:
        if params['classification']:
            gen_params = {'dim': params['data_shape'],
                          'shuffle': params['shuffle'],
                          'data_fmt': params['data_fmt']}
            train_x, train_y = load_categorical_data(list_IDs=addrs['train_addrs'],
                                                     labels=addrs['train_labels'],
                                                     **gen_params)
            val_x, val_y = load_categorical_data(list_IDs=addrs['val_addrs'],
                                                 labels=addrs['val_labels'],
                                                 **gen_params)
        else:  # regression
            gen_params = {'dim': params['data_shape'],
                          'shuffle': params['shuffle'],
                          'data_fmt': params['data_fmt'],
                          'regression_target': params['regression_target'],
                          'target_normalization': params['target_normalization']
                          }
            train_x, train_y = load_regression_data(list_IDs=addrs['train_addrs'],
                                                    dataset_path=params['regression_path'],
                                                    **gen_params)
            val_x, val_y = load_regression_data(list_IDs=addrs['val_addrs'],
                                                dataset_path=params['regression_path'],
                                                **gen_params)
    # Compile model
    opt_params = {'lr': params['lr'],
                  'beta_1': params['beta_1'],
                  'beta_2': params['beta_2'],
                  'decay': params['decay']}
    _optimizer = optimizers.Adam(**opt_params)
    if params['classification']:
        model.compile(optimizer=_optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=_optimizer,
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    # Create callbacks dependent on generator use
    callbacks = []
    if params['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=params['patience'],
                                       verbose=params['verbose'],
                                       mode='min'))
    if params['use_generator']:
        metrics = GeneratorClassificationMetrics(val_generator, val_steps)
        callbacks.append(metrics)
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=params['epochs'],
                                      validation_data=val_generator,
                                      validation_steps=val_steps,
                                      callbacks=callbacks,
                                      use_multiprocessing=False,
                                      verbose=params['verbose'])
    else:
        if params['classification']:
            metrics = ClassificationMetrics()
        else:  # regression
            metrics = RegressionMetrics()
        callbacks.append(metrics)
        history = model.fit(train_x, train_y,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            validation_data=(val_x, val_y),
                            callbacks=callbacks,
                            verbose=params['verbose'])

    model.save(os.path.join(params['out_dir'], "{}.h5".format(params['run_name'])))
    with open(os.path.join(params['out_dir'], "{}_history.pickle".format(params['run_name'])), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    if params['classification']:
        tracked_metrics = {'val_f1s': metrics.val_f1s,
                           'val_recalls': metrics.val_recalls,
                           'val_precisions': metrics.val_precisions}
    else:  # regression
        tracked_metrics = {'val_mse': metrics.mse,
                           'val_r2': metrics.r2}
    with open(os.path.join(params['out_dir'], "{}_metrics.pickle".format(params['run_name'])), 'wb') as file_pi:
        pickle.dump(tracked_metrics, file_pi)
    if params['classification']:
        training_results = {'val_loss': history.history['val_loss'][-1],
                            'val_acc': history.history['val_acc'][-1],
                            'val_f1': metrics.val_f1s[-1],
                            'val_recall': metrics.val_recalls[-1],
                            'val_precision': metrics.val_precisions[-1]}
    else:  # regression
        training_results = {'val_loss': history.history['val_loss'][-1],
                            'val_mse': -1 * metrics.mse[-1],
                            'val_r2': metrics.r2[-1]}
        training_results['default'] = training_results['val_r2']

    if params['verbose']:
        print("Time to complete run {}: {}".format(params['run_name'],
                                                   time.time() - start_time))
        print("Results:")
        for key in training_results:
            print("{} : {}".format(key, training_results[key]))

    reset_keras()
    return training_results


def training(params):
    default_results = {'val_loss': 0.,
                       'val_acc': 0.,
                       'val_f1': 0.,
                       'val_recall': 0.,
                       'val_precision': 0.,
                       'val_r2': 0.,
                       'val_mse': 0.
                       }
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
