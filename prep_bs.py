import numpy as np
import os
import load_mnist


def reshape_image(x):
    x = np.transpose(x,[0,2,3,1])
    return x

def fashion_label_change(x):
    for i in range(x.shape[0]):
       	x[i]+=10
    return x
def mnist_funct():
	mnist = dict()
	mnist['train_x'],mnist['train_y'],mnist['test_x'],mnist['test_y'] = load_mnist.load_mnist_dataset(os.getcwd()+"/mnist")
	mnist['train_x'] = reshape_image(mnist['train_x']/255)
	mnist['test_x'] = reshape_image(mnist['test_x']/255)
	return mnist

def fashion_funct():
	fashion = dict()
	fashion['train_x'],fashion['train_y'],fashion['test_x'],fashion['test_y'] = load_mnist.load_mnist_dataset(os.getcwd()+"/fashion")
	fashion['train_x'] = reshape_image(fashion['train_x']/255)
	fashion['test_x'] = reshape_image(fashion['test_x']/255)
	#label change
	fashion['train_y'] = fashion_label_change(fashion['train_y'])
	fashion['test_y'] = fashion_label_change(fashion['test_y'])
	print(np.max(fashion['test_y']),np.min(fashion['test_y']))
	print(np.max(fashion['train_y']),np.min(fashion['train_y']))
	return fashion

def merger():
	mnist = mnist_funct()
	fashion = fashion_funct()
	merged = dict()
	merged['train_x'] = np.vstack((mnist['train_x'],fashion['train_x']))
	merged['train_y'] = np.hstack((mnist['train_y'],fashion['train_y']))
	merged['test_x']  = np.vstack((mnist['test_x'],fashion['test_x']))
	merged['test_y']  = np.hstack((mnist['test_y'],fashion['test_y']))
	"""
	shuffle_train = np.random.permutation(merged['train_x'].shape[0])
	shuffle_test  = np.random.permutation(merged['test_x'].shape[0])
	np.take(merged['train_x'],shuffle_train,axis=0,out=merged['train_x'])
	np.take(merged['train_y'],shuffle_train,axis=0,out=merged['train_y'])
	np.take(merged['test_x'],shuffle_test,axis=0,out=merged['test_x'])
	np.take(merged['test_y'],shuffle_test,axis=0,out=merged['test_y'])
	"""
	return merged


	
	





