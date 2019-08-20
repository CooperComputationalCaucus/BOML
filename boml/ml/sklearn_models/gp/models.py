"""
@author: maffettone

Gaussian process models for classification and regression
"""

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


def _kernel(params):
    if params['nu'] == float('inf'):
        return RBF(length_scale=params['length_scale'])
    else:
        return Matern(length_scale=params['length_scale'],
                      nu=params['nu'])


def gen_classifier(params):
    clf = GaussianProcessClassifier(kernel=_kernel(params),
                                    n_jobs=params['n_jobs'],
                                    copy_X_train=False
                                    )
    return clf


def gen_regressor(params):
    clf = GaussianProcessRegressor(kernel=_kernel(params),
                                   normalize_y=True,
                                   copy_X_train=False
                                   )
    return clf


def gen_model(params):
    if params['classification']:
        return gen_classifier(params)
    elif params['regression']:
        return gen_regressor(params)
    else:
        raise KeyError('Neither regression or classification specified!')
