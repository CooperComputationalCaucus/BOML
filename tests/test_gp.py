"""
@author: maffettone

Quick test of gp models
"""

if __name__ == '__main__':
    import sys

    sys.path.append('../')
    from boml.optimization import optimize_nets

    config = {'verbose': 2,
              'max_iter': 3,
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
              'architecture': 'gp',
              'variable_space': {
                  'log_length_scale': [-1, 0.5, 0.0001],
              },
              'fixed_space': {
              },
              'training_params': {
                  'cv': 5,
                  'dataset_dir': '../test_data/organic_rgb/',
                  'out_dir': '../test_data',
                  'run_name': 'test',
                  'batch_size': 8,
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
                  'architecture': 'gp',
                  'variable_space': {
                      'log_length_scale': [-1, 0.5, 0.0001],
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
                      'batch_size': 8,
                      'data_shape': (3072,),
                      'data_fmt': 'png',
                      'verbose': True
                  }
                  }

    optimize_nets(config_dict=config)
    optimize_nets(config_dict=reg_config)