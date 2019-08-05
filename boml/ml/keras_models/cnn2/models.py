"""
@author: pmm

These are a set of convolutional neural net models that take the form:
[Convolution--Convolution--(Dropout)--Max Pooling--]-- Flatten -- [Dense---Activation--(Dropout)--]--Batch Normalization
for classification and regression problems. Repeat units are in [] and optional units are in ().

The dense completion is contained in boml.ml.keras_models.cnn.models
"""

from keras import models
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Conv3D, MaxPooling3D

from boml.ml.keras_models.cnn.models import complete_model


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
                     activation='relu',
                     input_shape=params['data_shape']
                     ))
    model.add(Conv1D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     activation='relu'
                     ))
    model.add(Dropout(rate=params['conv_dropout']))
    model.add(MaxPooling1D(pool_size=params['pool_sizes'][0],
                           strides=None,
                           padding='same'))
    for idx in range(1, params['conv_units']):
        model.add(Conv1D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same',
                         activation='relu'
                         ))
        model.add(Conv1D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same',
                         activation='relu'
                         ))
        model.add(Dropout(rate=params['conv_dropout']))
        model.add(MaxPooling1D(pool_size=params['pool_sizes'][idx],
                               strides=None,
                               padding='same'))
    model = complete_model(params, model)
    return model


def gen_2D_model(params):
    """
    Generates and returns model according to parameters from parameters.gen_hyperparameters()
    """
    model = models.Sequential()
    model.add(Conv2D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     activation='relu',
                     input_shape=params['data_shape']
                     ))
    model.add(Conv2D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     activation='relu'
                     ))
    model.add(Dropout(rate=params['conv_dropout']))
    model.add(MaxPooling2D(pool_size=params['pool_sizes'][0],
                           strides=None,
                           padding='same'))
    for idx in range(1, params['conv_units']):
        model.add(Conv2D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same',
                         activation='relu'
                         ))
        model.add(Conv2D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same',
                         activation='relu'
                         ))
        model.add(Dropout(rate=params['conv_dropout']))
        model.add(MaxPooling2D(pool_size=params['pool_sizes'][idx],
                               strides=None,
                               padding='same'))
    model = complete_model(params, model)
    return model


def gen_3D_model(params):
    """
    Generates and returns model according to parameters from parameters.gen_hyperparameters()
    """
    model = models.Sequential()
    model.add(Conv3D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     activation='relu',
                     input_shape=params['data_shape']
                     ))
    model.add(Conv3D(filters=params['conv_filters'][0],
                     kernel_size=params['conv_kernels'][0],
                     strides=params['conv_strides'][0],
                     padding='same',
                     activation='relu'
                     ))
    model.add(Dropout(rate=params['conv_dropout']))
    model.add(MaxPooling3D(pool_size=params['pool_sizes'][0],
                           strides=None,
                           padding='same'))
    for idx in range(1, params['conv_units']):
        model.add(Conv3D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same',
                         activation='relu'
                         ))
        model.add(Conv3D(filters=params['conv_filters'][idx],
                         kernel_size=params['conv_kernels'][idx],
                         strides=params['conv_strides'][idx],
                         padding='same',
                         activation='relu'
                         ))
        model.add(Dropout(rate=params['conv_dropout']))
        model.add(MaxPooling3D(pool_size=params['pool_sizes'][idx],
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