"""
@author: maffettone

Gradient boosting models for classification and regression
"""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def gen_classifier(params):
    clf = GradientBoostingClassifier(learning_rate=params['learning_rate'],
                                     n_estimators=params['n_estimators'],
                                     min_samples_split=params['min_samples_split'],
                                     max_features=params['max_features'])
    return clf


def gen_regressor(params):
    clf = GradientBoostingRegressor(learning_rate=params['learning_rate'],
                                    n_estimators=params['n_estimators'],
                                    min_samples_split=params['min_samples_split'],
                                    max_features=params['max_features'])
    return clf


def gen_model(params):
    if params['classification']:
        return gen_classifier(params)
    elif params['regression']:
        return gen_regressor(params)
    else:
        raise KeyError('Neither regression or classification specified!')
