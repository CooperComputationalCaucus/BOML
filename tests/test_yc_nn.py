# TODO: Write and run tests of regression and classification for general and
#  specific nn
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from boml.optimization import optimize_nets

    optimize_nets(architecture='nn_yc')
