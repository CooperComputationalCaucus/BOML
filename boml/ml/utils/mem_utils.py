'''
Created on 7 May 2019

@author: pmm
'''
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
import numpy as np
import gc

GPU_RAM_LIMIT = 11
CPU_RAM_LIMIT = 32


def reset_keras(*args):
    # sess=get_session()
    # clear_session()
    # sess.close()
    # sess = get_session()
    K.clear_session()
    try:
        for arg in args:
            del arg
    except:
        pass
    gc.collect()

def config_keras():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8 #Use fixed fraction of GPU memory
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  
    

def check_model_memory(model,batch_size):
    '''
    Checks viability for GPU ram requirements (conservatively). 
    
    Arguments
    ---------
        model: keras/tf model
        batch_size: integer, batch size
        
    Raises
    ------
    Value Error: if expected memory requirements exceed global settings 
    '''
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    total_memory += number_size*3*(trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    if gbytes>GPU_RAM_LIMIT:
        raise ValueError("Model is too large. Estimated memory required is {} GB".format(gbytes))

def check_dataset_memory(m,data_shape,batch_size=1,use_generator=False):
    '''
    Checks viability for CPU ram requirements of dataset. 
    
    Arguments
    ---------
        m: integer, number of points in dataset
        data_shape: tuple, shape of input data
        batch_size: integer, batch size
        use_generator: logical, from training parameters use of generator vs complete dataset
        
    Raises
    ------
        Value Error: if expected memory requirements exceed global settings 
    '''
    from functools import reduce
    cpu_ram = CPU_RAM_LIMIT
    if use_generator:
        mem_req_data = 4*2*reduce(lambda x,y: x*y, data_shape)*batch_size/10**9  
    else:
        mem_req_data = 4*2*reduce(lambda x,y: x*y, data_shape)*m/10**9   
    if(0.8*cpu_ram<mem_req_data):
        print(mem_req_data,'GB required') 
        raise ValueError("The amount of memory required to load the dataset exceeds 80% the available "\
                         "RAM. Please reduce the oversampling, or use a generator based training")
