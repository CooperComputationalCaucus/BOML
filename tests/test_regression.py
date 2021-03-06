"""
@author: maffettone

Quick test of regression models in CNN
"""

if __name__=='__main__':
    import sys
    sys.path.append('../')
    from boml.optimization import optimize_nets
    config = {'verbose': 2,
              'debug_msgs': False,
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
              'variable_space': {
                  'conv_units': [2, 6, 1],
                  'filters': [10, 100, 1],
                  'filters_roc': [-1.0, 1.0, 0.1],
                  'kernel_size': [2, 10, 1],
                  'kernel_roc': [-1.5, 1.0, 0.1],
                  'stride': [1, 3, 1],
                  'stride_roc': [-1.5, 1.0, 0.1],
                  'conv_dropout': [0.0, 0.1, 0.001],
                  'pool_size': [1, 4, 1],
                  'pool_roc': [-1.5, 0., .1],
                  'dense_units': [0, 2, 1],
                  'nodes': [10, 1000, 10],
                  'nodes_roc': [-2.0, 0.0, 0.1],
                  'dense_dropout': [0.0, 0.8, 0.001]
              },
              'fixed_space': {
              },
              'training_params': {
                  'val_split': 0.2,
                  'dataset_dir': '../test_data/organic_rgb/0/',
                  'regression_path': '../test_data/organic_rgb/cat0_random_Energy.csv',
                  'regression_target': 'Energy',
                  'target_normalization': True,
                  'early_stopping': False,
                  'patience': 4,
                  'epochs': 2,
                  'out_dir': '../test_data',
                  'run_name': 'test',
                  'batch_size': 8,
                  'data_shape': (32, 32, 3),
                  'verbose': True,
                  'use_generator': False
              }
              }
    optimize_nets(config_dict=config)
