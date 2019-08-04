"""
@author: pmm

These are a set of convolutional neural net models that take the form:
[Convolution--Activation--(Dropout)--Avgerage Pooling--] -- Flatten -- [Dense---Activation--(Dropout)--]
for classification problems. Repeat units are in [] and optional units are in ().
"""
from keras import models
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, AveragePooling2D, Conv1D, AveragePooling1D, Conv3D, AveragePooling3D

from boml.ml.utils.mem_utils import check_model_memory


def complete_model(params, model):
    """
    Comlpetes the model development by flattening, adding dense layers, and
    finishing for regression or classification
    Parameters
    ----------
    params: dictionary of standard parameters
    model: Keras model

    Returns
    -------
    model: Keras model
    """

    model.add(Flatten())
    for idx in range(params['dense_units']):
        model.add(Dense(params['dense_dims'][idx]))
        model.add(Activation('relu'))
        model.add(Dropout(rate=params['dense_dropout']))

    if params['classification']:
        model.add(Dense(params['n_classes']))
        model.add(Activation('softmax'))
    elif params['regression']:
        model.add(Dense(1))
        model.add(Activation('linear'))
    else:
        raise ValueError('Model is not set for regression nor classification')

    if params['verbose']:
        model.summary()
    if params['verbose']:
        for key, val in params.items():
            print(key, ' : ', val)
    check_model_memory(model, params['batch_size'])
    return model


# noinspection PyPep8Naming
def gen_1D_model(params):
    """
    Generates and returns model according to parameters from parameters.gen_hyperparameters()
    """
    model = models.Sequential()
    model.add(Conv1D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     input_shape=params['data_shape']
                     ))
    model.add(Activation('relu'))
    model.add(Dropout(rate=params['conv_dropout']))
    model.add(AveragePooling1D(pool_size=params['pool_sizes'][0],
                               strides=None,
                               padding='same'))
    for idx in range(1, params['conv_units']):
        model.add(Conv1D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same'
                         ))
        model.add(Activation('relu'))
        model.add(Dropout(rate=params['conv_dropout']))
        model.add(AveragePooling1D(pool_size=params['pool_sizes'][idx],
                                   strides=None,
                                   padding='same'))
    model = complete_model(params, model)
    return model


# noinspection PyPep8Naming
def gen_2D_model(params):
    """
    Generates and returns model according to parameters from parameters.gen_hyperparameters()
    """
    model = models.Sequential()
    model.add(Conv2D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     input_shape=params['data_shape']
                     ))
    model.add(Activation('relu'))
    model.add(Dropout(rate=params['conv_dropout']))
    model.add(AveragePooling2D(pool_size=params['pool_sizes'][0],
                               strides=None,
                               padding='same'))
    for idx in range(1, params['conv_units']):
        model.add(Conv2D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same'
                         ))
        model.add(Activation('relu'))
        model.add(Dropout(rate=params['conv_dropout']))
        model.add(AveragePooling2D(pool_size=params['pool_sizes'][idx],
                                   strides=None,
                                   padding='same'))

    model = complete_model(params, model)
    return model


# noinspection PyPep8Naming
def gen_3D_model(params):
    """
    Generates and returns model according to parameters from parameters.gen_hyperparameters()
    """
    model = models.Sequential()
    model.add(Conv3D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     input_shape=params['data_shape']
                     ))
    model.add(Activation('relu'))
    model.add(Dropout(rate=params['conv_dropout']))
    model.add(AveragePooling3D(pool_size=params['pool_sizes'][0],
                               strides=None,
                               padding='same'))
    for idx in range(1, params['conv_units']):
        model.add(Conv3D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same'
                         ))
        model.add(Activation('relu'))
        model.add(Dropout(rate=params['conv_dropout']))
        model.add(AveragePooling3D(pool_size=params['pool_sizes'][idx],
                                   strides=None,
                                   padding='same'))

    model = complete_model(params, model)
    return model


def gen_model(params):
    if len(params['data_shape']) == 2:
        return gen_1D_model(params)
    elif len(params['data_shape']) == 3:
        return gen_2D_model(params)
    elif len(params['data_shape']) == 4:
        return gen_3D_model(params)
    else:
        raise IndexError('Input data shape of {} inappropriate for model building'.format(params['data_shape']))
