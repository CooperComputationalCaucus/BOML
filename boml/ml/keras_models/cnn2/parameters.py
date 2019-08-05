"""
@author: pmm
Configuration parameters are global and for the optimizer.
Metaparameters are used to generate hyperparameters (for instance by making logarithmic searching easy).
Hyperparameters are passed to a given training run.
As such, there are many shared key:value pairs between the metaparameter and hyperparameter dictionaries.

Much is borrowed from singular CNN models
"""

from boml.ml.keras_models.cnn.parameters import load_metaparameters as lm
from boml.ml.keras_models.cnn.parameters import gen_hyperparameters as gh


def load_metaparameters(param_dict=None):
    """
    Parameters for bayesian optimizer. Default dictionary listed and updated by param_dict.
    """
    return lm(param_dict)


def gen_hyperparameters(metaparams):
    """
    Parameters for model architecture and training. Default dictionary listed and updated by metaparameters.

    First metaparameters required are used to assemble lists describing the architecture.
    Then metaparameters not required are used to update the dictionary below.
    """
    return gh(metaparams)


if __name__ == '__main__':
    metaparams = load_metaparameters()
    params = gen_hyperparameters(metaparams)
    for key in params:
        print(key, params[key])
