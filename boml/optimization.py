import json
import glob
import pickle

from boml.bayes_opt import DiscreteBayesianOptimization
from boml.bayes_opt.event import Events
from boml.bayes_opt import UtilityFunction
from boml.utils.defaults import default_config
from boml.utils.sanity_checks import sanity_checks

import time
import sys


class Optimizer:

    def __init__(self, config):
        self.start_time = time.time()
        self.config = config
        self.dbo = DiscreteBayesianOptimization(f=None,
                                                prange=config['variable_space'],
                                                verbose=config['verbose'],
                                                random_state=config['seed'],
                                                init_points=config['init_random'],
                                                feature_scaling=config['feature_scaling'])
        self.utility = UtilityFunction(kind='ucb', kappa=2.5, xi=0.0)
        model_functions = self.fetch_model_functions()
        self.load_metaparameters = model_functions['load_metaparameters']
        self.gen_hyperparameters = model_functions['gen_hyperparameters']
        self.training = model_functions['training']
        self.iter_count = 0

    def fetch_model_functions(self):
        """
        Method to load necessary packages, and pass back standard functions
        Must be updated for each new package in boml.ml

        Returns
        =======
        model_functions, dictionary with keys:
            load_metaparameters
            gen_hyperparameters
            training

        """
        if self.config['architecture'] == 'cnn':
            from boml.ml.keras_models.cnn import load_metaparameters, gen_hyperparameters
            from boml.ml.keras_models.training import training
        # elif self.config['architecture'] == 'cnn2':
        #     from .ml.cnn2.parameters import load_metaparameters, gen_hyperparameters
        #     from .ml.cnn2.models import training
        # elif self.config['architecture'][0:2] == 'nn':
        #     from .ml.nn.parameters import load_metaparameters, gen_hyperparameters
        #     from .ml.nn.models import training
        # elif self.config['architecture'] == 'rf':
        #     from .ml.shallow.parameters import load_metaparameters, gen_hyperparameters
        #     from .ml.shallow.models import training
        # elif self.config['architecture'] == 'nb':
        #     from .ml.shallow.parameters import load_metaparameters, gen_hyperparameters
        #     from .ml.shallow.models import training
        # elif self.config['architecture'] == 'svm':
        #     from .ml.shallow.parameters import load_metaparameters, gen_hyperparameters
        #     from .ml.shallow.models import training
        # elif self.config['architecture'] == 'gp':
        #     from .ml.shallow.parameters import load_metaparameters, gen_hyperparameters
        #     from .ml.shallow.models import training
        else:
            load_metaparameters = None
            gen_hyperparameters = None
            training = None

        model_functions = {'load_metaparameters': load_metaparameters,
                           'gen_hyperparameters': gen_hyperparameters,
                           'training': training}
        return model_functions

    def priming(self):
        config = self.config
        if config['verbose']:
            self.dbo._prime_subscriptions()
            self.dbo.dispatch(Events.OPTMIZATION_START)

        if config['previous_points']:
            if config['debug_msgs']:
                print("Adding previous points...")
                sys.stdout.flush()
            points = []
            fnames = sorted(
                glob.glob("{}/{}_*_metaparams".format(config['training_params']['out_dir'], config['basename'])))
            for fname in fnames:
                with open(fname, 'rb') as f: points.append(pickle.load(f))  # 2-tuple of dictionary and scalar
            for point in points:
                self.dbo.register(params=point[0], target=point[1])
            if config['debug_msgs']:
                print("Finished adding previous points...")
                sys.stdout.flush()

    def run_loop(self):
        config = self.config
        meta_dict = {}
        for idx in range(config['max_iter']):
            if config['debug_msgs']:
                debug_start = time.time()
                print("Searching for new point...", end=' ')
                sys.stdout.flush()
            next_point = \
                self.dbo.suggest(self.utility, sampler=config['sampler'], multiprocessing=config['multiprocessing'])[0]
            # Setup parameters
            meta_dict.update(next_point)
            meta_dict.update(config['fixed_space'])
            metaparams = self.load_metaparameters(meta_dict)
            metaparams.update(config['training_params'])
            params = self.gen_hyperparameters(metaparams)
            params['run_name'] = "{}_{:04d}".format(config['basename'], config['state'] + idx)
            params['seed'] = config['seed']
            params['architecture'] = config['architecture']
            params['regression'] = config['regression']
            params['classification'] = config['classification']

            # Train net
            if config['debug_msgs']:
                print("Search took {} min...".format((time.time() - debug_start) / 60), end=' ')
                debug_start = time.time()
                print("Training new network...", end=' ')
                sys.stdout.flush()
            training_results = self.training(params)
            target = training_results[config['target']]

            # Register and save result
            if config['debug_msgs']:
                print("Training took {} min...".format((time.time() - debug_start) / 60), end=' ')
                debug_start = time.time()
                print("Register and saving outputs...")
                sys.stdout.flush()
            self.dbo.register(params=next_point, target=target)
            fname = "{}/{}_{:04d}_metaparams.pickle".format(config['training_params']['out_dir'],
                                                            config['basename'],
                                                            config['state'] + idx)
            with open(fname, 'wb') as f:
                pickle.dump((next_point, target), f, protocol=pickle.HIGHEST_PROTOCOL)
            fname = "{}/{}_{:04d}_hyperparams.json".format(config['training_params']['out_dir'],
                                                           config['basename'],
                                                           config['state'] + idx)
            with open(fname, 'w') as f:
                json.dump(params, f, indent=4)
            self.iter_count += 1
            if time.time() - self.start_time > config['max_time']:
                print("Time limit exceeded!")
                break
        return True

    def ending(self):
        self.dbo.dispatch(Events.OPTMIZATION_END)
        print("Maximum value found using discrete GPs: {}".format(self.dbo.max['target']))
        print("Time taken: {} seconds for {} iterations".format(time.time() - self.start_time, self.iter_count))


def optimize_nets(config_file=None, config_dict=None):
    '''
    Optimizer loop for loading configuration and running multiple architectures/hyperparameters 
    in search for optimal score. 
    '''

    if config_file:
        with open(config_file, 'r') as f:
            config_inp = json.load(f)
        try:
            config = default_config(config_inp['architecture'])
        except KeyError:
            config = default_config()
        config.update(config_inp)
    else:
        config = default_config()
    if config_dict:
        config.update(config_dict)
    sanity_checks(config)

    optimizer = Optimizer(config)
    optimizer.priming()
    optimizer.run_loop()
    optimizer.ending()


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    optimize_nets()
