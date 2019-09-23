#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yu, pmm

"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from boml.ml.utils.mem_utils import check_model_memory
import warnings


def complete_model(params, model):
    """
    Completes by finishing for regression or classification
    Parameters
    ----------
    params: dictionary of standard parameters
    model: Keras model

    Returns
    -------
    model: Keras model
    """
    if params['classification']:
        model.add(Dense(params['n_classes']))
        model.add(Activation('softmax'))
    elif params['regression']:
        model.add(Dense(1))
        model.add(Activation('linear'))
    else:
        raise ValueError(
            'Model is not set for regression nor classification'
        )

    if params['verbose']:
        model.summary()
    if params['verbose']:
        for key, val in params.items():
            print(key, ' : ', val)
    check_model_memory(model, params['batch_size'])
    return model


def gen_nn_general(params):
    """
    Generates a general neural net
    with a variable number of hidden layers and decay rate
    """
    model = Sequential()
    model.add(Flatten(input_shape=params['data_shape']))
    for dim in params['dense_dims']:
        model.add(Dense(dim))
        model.add(Activation('relu'))
        model.add(Dropout(rate=params['dense_dropout']))
    if params['dense_batchnorm']:
        model.add(BatchNormalization())
    model = complete_model(params, model)
    return model


def gen_nn2(params):
    """
    Generates a deep neural net with a 2 hidden layers
    """
    assert len(params['dense_dims']) >= 2
    if len(params['dense_dims']) > 2:
        warnings.warn('Model is using 2 hidden layers, '
                      'but received more than 2 dense dimensions.', Warning)
    new_params = params
    new_params['dense_dims'] = new_params['dense_dims'][:2]
    return gen_nn_general(new_params)


def gen_nn3(params):
    """
    Generates a deep neural net with a 3 hidden layers
    """
    assert len(params['dense_dims']) >= 3
    if len(params['dense_dims']) > 3:
        warnings.warn('Model is using 3 hidden layers, '
                      'but received more than 3 dense dimensions.', Warning)
    new_params = params
    new_params['dense_dims'] = new_params['dense_dims'][:3]
    return gen_nn_general(new_params)


def gen_nn4(params):
    """
    Generates a deep neural net with a 4 hidden layers
    """
    assert len(params['dense_dims']) >= 4
    if len(params['dense_dims']) > 4:
        warnings.warn('Model is using 4 hidden layers, '
                      'but received more than 4 dense dimensions.', Warning)
    new_params = params
    new_params['dense_dims'] = new_params['dense_dims'][:4]
    return gen_nn_general(new_params)


def gen_model(params):
    """
    Generates neural net with default 3 hidden layers
    """
    if params['architecture'] == 'nn_yc':
        return gen_nn_general(params)
    elif params['architecture'] == 'nn_general':
        return gen_nn_general(params)
    elif params['architecture'] == 'nn2':
        return gen_nn2(params)
    elif params['architecture'] == 'nn3':
        return gen_nn3(params)
    elif params['architecture'] == 'nn4':
        return gen_nn4(params)
    else:
        raise ValueError(
            "Architecture type {} unavailable.".format(params['architecture'])
        )
