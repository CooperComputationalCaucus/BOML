"""
@author: maffettone

SVM models for classification and regression
"""

from sklearn.svm import SVC
from sklearn.svm import SVR


def gen_classifier(params):
    clf = SVC(probability=False,
              C=params['C'],
              gamma=params['gamma']
              )
    return clf


def gen_regressor(params):
    clf = SVR(C=params['C'],
              gamma=params['gamma']
              )
    return clf


def gen_model(params):
    if params['classification']:
        return gen_classifier(params)
    elif params['regression']:
        return gen_regressor(params)
    else:
        raise KeyError('Neither regression or classification specified!')
