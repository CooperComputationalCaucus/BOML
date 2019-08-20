"""
@author: maffettone

Quick test of SVM models
"""

if __name__ == '__main__':
    import sys

    sys.path.append('../')
    from boml.optimization import optimize_nets

    config = {'verbose': 2,
              'max_iter': 5,
              'max_time': 3600,
              'init_random': 2,
              'seed': None,
              'state': 1,
              'sampler': 'greedy',
              'basename': 'test',
              'target': 'val_f1',
              'multiprocessing': 1,
              'previous_points': False,
              'feature_scaling': False,
              'regression': False,
              'classification': True,
              'architecture': 'rf',
              'variable_space': {
                  'log_n_estimators': [0, 3, 0.1],
                  'min_samples_split': [2, 3, 1],
                  'max_features': [0.1, 0.9, 0.001]
              },
              'fixed_space': {
              },
              'training_params': {
                  'cv': 5,
                  'dataset_dir': '../test_data/organic_rgb/',
                  'out_dir': '../test_data',
                  'run_name': 'test',
                  'data_shape': (3072,),
                  'data_fmt': 'png',
                  'verbose': True
              }
              }

    reg_config = {'verbose': 2,
                  'max_iter': 3,
                  'max_time': 3600,
                  'init_random': 2,
                  'seed': None,
                  'state': 1,
                  'sampler': 'greedy',
                  'basename': 'test',
                  'target': 'val_r2',
                  'multiprocessing': 1,
                  'previous_points': False,
                  'feature_scaling': False,
                  'regression': True,
                  'classification': False,
                  'architecture': 'svm',
                  'variable_space': {
                      'log_n_estimators': [0, 3, 0.1],
                      'min_samples_split': [2, 3, 1],
                      'max_features': [0.1, 0.9, 0.001]
                  },
                  'fixed_space': {
                  },
                  'training_params': {
                      'cv': 5,
                      'dataset_dir': '../test_data/organic_rgb/0/',
                      'regression_path': '../test_data/organic_rgb/cat0_random_Energy.csv',
                      'regression_target': 'Energy',
                      'target_normalization': True,
                      'out_dir': '../test_data',
                      'run_name': 'test',
                      'data_shape': (3072,),
                      'data_fmt': 'png',
                      'verbose': True
                  }
                  }

    optimize_nets(config_dict=config)
    optimize_nets(config_dict=reg_config)
