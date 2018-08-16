import os
import struct
import fnmatch
import gzip
import numpy as np

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist_dataset(path=os.getcwd()+"/mnist"):

    pattern = '*ubyte*'

    # crawl directory and grab filenames
    names = []
    for path, subdirs, files in os.walk(path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                names.append(os.path.join(path, filename))
                
    num_files = len(names)

    #print(names)   
    # read the files into a numpy array
    data = dict()
    data['train_imgs'] = None
    data['test_imgs'] = None
    data['train_labels'] = None
    data['test_labels'] = None
    for i in range(num_files):
        if 'train' in names[i]:
            if 'images' in names[i]:
                data['train_imgs'] = read_idx(names[i]) 
            else:
                data['train_labels'] = read_idx(names[i])
        else:
            if 'images' in names[i]:
                data['test_imgs'] = read_idx(names[i])
            else:
                data['test_labels'] = read_idx(names[i])

    X_train = data['train_imgs']
    X_test = data['test_imgs']
    y_train = data['train_labels']
    y_test = data['test_labels']

    data['train_imgs'] = X_train = X_train[:,np.newaxis].astype(np.float32)
    data['test_imgs'] = X_test = X_test[:,np.newaxis].astype(np.float32)

    #return data
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
	tr_x,tr_y,ts_x,ts_y = load_data("C:/Users/k_tej/Documents/TEJA/ML_resources/ML_concepts/CNN/my_cnn_test_models/MNIST_data")
	print(tr_x.shape)
	print(ts_x.shape)
