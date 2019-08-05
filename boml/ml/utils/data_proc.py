'''
Created on 16 Apr 2019

@author: pmm
'''
import os
import glob
from random import shuffle,seed
from keras.utils import Sequence, to_categorical
import numpy as np
import imageio
import pandas as pd

def ListAddressesLabels(params):
    '''
    Develops a list of addresses to be fed to a generator. 
    Arguments
    ---------
    dataset_directory is the main directory
    dataset format is 'categorical_folders' and has files in folders corresponding to Classifier
    
    params is a dictionary of parameters requiring:
        'val_split'
        'dataset_dir' 
        'shuffle' 
        'use_generator'
        'data_shape'
    Returns
    -------
    data : {
        train_addrs
        train_labels
        val_addrs
        val_labels}
        Where labels will vary by dataset_fmt
    '''
    if 'seed' in params and params['seed']:
        seed(params['seed'])
    
    dataset_dir = params['dataset_dir']
    fmt = params['data_fmt']
    #Develop list of addresses
    tmp = os.listdir(dataset_dir)
    categories = []
    for cat in tmp: 
        if os.path.isdir(os.path.join(dataset_dir,cat)): 
            categories.append(cat)
    
    train_addrs = []
    train_labels = []
    val_addrs = []
    val_labels = []
    train_split = 1 - params['val_split']
    
    # Empty list just loads all files in top directory
    if not categories:
        categories.append('./')
        
    for cat in categories:
        gen_path = os.path.join(dataset_dir,cat)+'/*.{}'.format(fmt)
        cat_paths = glob.glob(gen_path)
        if params['shuffle']: shuffle(cat_paths)
        train_addrs.extend(cat_paths[0:int(train_split*len(cat_paths))])
        val_addrs.extend(cat_paths[int(train_split*len(cat_paths)):])
        if cat == './':
            train_labels.extend([0 for _ in range(len(cat_paths[0:int(train_split*len(cat_paths))]))])  
            val_labels.extend([0 for _ in range(len(cat_paths[int(train_split*len(cat_paths)):]))])
        else:
            train_labels.extend([int(cat) for _ in range(len(cat_paths[0:int(train_split*len(cat_paths))]))])  
            val_labels.extend([int(cat) for _ in range(len(cat_paths[int(train_split*len(cat_paths)):]))])
        
    data = {}
    data['train_addrs'] = train_addrs
    data['train_labels'] = train_labels
    data['val_addrs'] = val_addrs
    data['val_labels'] = val_labels
    return data

def default_preprocessing(x):
    return x

def load_categorical_data(list_IDs, labels, dim=(2048,), data_fmt='png',
                          shuffle=True, one_hot=True, preprocessing = default_preprocessing):
    '''
    Loading data from a list of file IDs (full paths)
    
    Arguments
    ---------
    list_IDs : list of full path strings
    labels : list of string or integer labels
    dim : data dimensionality, perfoms a reshape and ravel by default
    shuffle : logical
    one_hot : logical, to convert labels as is to one_hot vectors
    preprocessing : function, takes in x value and returns a processed x value
    
    Returns
    -------
    X : data
    y : labels
    '''
    X = np.empty((len(list_IDs),*dim))
    if shuffle:
        comb = list(zip(list_IDs,labels))
        np.random.shuffle(comb)
        list_IDs[:], labels[:] = zip(*comb)
    
    if data_fmt == 'npy':
        for i, ID in enumerate(list_IDs):
            # Store sample
            X[i,] = preprocessing(np.load(ID).reshape(dim))
    elif data_fmt == 'png':
        for i, ID in enumerate(list_IDs):
            # Store sample
            X[i,] = preprocessing(np.array(imageio.imread(ID)[...,0:3]).reshape(dim))
    else:
        raise ValueError('Generator not equipped for {} data type'.format(data_fmt))
    y = labels
    if one_hot:
        return X, to_categorical(y)
    else:
        return X, [int(_y) for _y in y]
    
class CategoricalDataGenerator(Sequence):
    '''Generates labeled data for Keras'''
    
    def __init__(self, list_IDs, labels, batch_size=32, dim=(2048,), shuffle=True, data_fmt='png', 
                 one_hot=True, num_classes=10, preprocessing = default_preprocessing):
        '''
        Arguments
        ---------
        list_IDs : list of full path strings
        labels : list of string or integer labels
        batch_size : integer number of data per batch
        dim : data dimensionality, perfoms a reshape and ravel by default
        shuffle : logical
        one_hot : logical, to convert labels as is to one_hot vectors
        num_classes : integer number of classes required for generator not full dataset
        preprocessing: function, takes in x value and returns a processed x value
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs=[]
        self.list_IDs.extend(list_IDs)
        self.labels = []
        self.labels.extend(labels)
        self.shuffle = shuffle
        self.data_fmt = data_fmt
        self.one_hot = one_hot
        self.num_classes=num_classes
        self.preprocessing = preprocessing
        self.on_epoch_end()
        self.i = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        y = [self.labels[k] for k in indexes]
        
        # Generate data
        X = self._data_generation(list_IDs_temp)
        if self.one_hot:
            return X, to_categorical(y,num_classes=self.num_classes)
        else:
            return X, [int(_y) for _y in y]
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        # Generate data
        if self.data_fmt == 'npy':
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,] = self.preprocessing(np.load(ID).reshape(self.dim))
        elif self.data_fmt == 'png':
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,] = self.preprocessing(np.array(imageio.imread(ID)[...,0:3]).reshape(self.dim))
        else:
            raise ValueError('Generator not equipped for {} data type'.format(self.data_fmt))      
        return X

def load_regression_data(list_IDs, dataset_path, regression_target = 'default',dim=(2048,), target_normalization=False,
                    data_fmt='png', shuffle=True, preprocessing = default_preprocessing):
    #TODO: implement regression data generator
    '''
    Loading data from a list of file IDs (full paths)
    
    Arguments
    ---------
    list_IDs : list of full path strings
    labels : list of string or integer labels
    dim : data dimensionality, perfoms a reshape and ravel by default
    shuffle : logical
    one_hot : logical, to convert labels as is to one_hot vectors
    preprocessing : function, takes in x value and returns a processed x value
    
    Returns
    -------
    X : data
    y : fname : filename without extension for referencing 
    '''
    
    # Get regression target dataset
    if os.path.splitext(dataset_path)[1] =='.pickle':
        df = pd.read_pickle(dataset_path)
    elif os.path.splitext(dataset_path)[1] =='.csv':
        df = pd.read_csv(dataset_path)
        df.set_index('Name', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Load x data and file names
    X = np.empty((len(list_IDs),*dim))
    if shuffle:
        np.random.shuffle(list_IDs)
    
    if data_fmt == 'npy':
        for i, ID in enumerate(list_IDs):
            X[i,] = preprocessing(np.load(ID).reshape(dim))
    elif data_fmt == 'png':
        for i, ID in enumerate(list_IDs):
            X[i,] = preprocessing(np.array(imageio.imread(ID)[...,0:3]).reshape(dim))
    else:
        raise ValueError('Generator not equipped for {} data type'.format(data_fmt))
    
    # Get y data
    fnames = [os.path.splitext(os.path.basename(ID))[0][:-4] for ID in list_IDs]
    y = df.loc[fnames][regression_target]
    if target_normalization: y = (y-np.min(y))/(np.max(y)-np.min(y))
    
    return X,y.values
        
#### MAIN BODY FOR TESTING #######
if __name__ == '__main__':
    params ={'val_split': .1,
             'dataset_dir':'../test_data/ternary_Cr-Al-Co/notexture_datasets/',
             'shuffle':False,
             'use_generator':False,
             'data_shape' :(2048,)}
    data= ListAddressesLabels(params)
    print('List ID results')
    print(list(zip(data.keys(),map(len,data.values()))))
    print(data['train_addrs'][0], data['train_labels'][0])
    X,y = load_categorical_data(data['train_addrs'],data['train_labels'])
    print("\nLoad data results")
    print('X shape',X.shape)
    print('y shape',y.shape)
    train_generator = CategoricalDataGenerator(data['train_addrs'],data['train_labels'])
    print('\nGenerator results')
    X,y = train_generator.__getitem__(0)
    print('X shape',X.shape)
    print('y shape',y.shape)
    print("Num epochs = ",len(train_generator))
    