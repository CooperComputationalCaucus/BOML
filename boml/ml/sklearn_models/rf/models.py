"""
@author: maffettone

SVM models for classification and regression
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def gen_classifier(params):
    clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                 min_samples_split=params['min_samples_split'],
                                 max_features=params['max_features'],
                                 n_jobs=params['n_jobs'])
    return clf


def gen_regressor(params):
    clf = RandomForestRegressor(n_estimators=params['n_estimators'],
                                min_samples_split=params['min_samples_split'],
                                max_features=params['max_features'],
                                n_jobs=params['n_jobs'])
    return clf


def gen_model(params):
    if params['classification']:
        return gen_classifier(params)
    elif params['regression']:
        return gen_regressor(params)
    else:
        raise KeyError('Neither regression or classification specified!')
