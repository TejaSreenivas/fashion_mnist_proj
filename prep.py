import numpy as np
import os
import load_mnist
import tensorflow as tf
from math import ceil, floor
IMAGE_SIZE = 28
channels = 1
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
	shuffle_train = np.random.permutation(merged['train_x'].shape[0])
	shuffle_test  = np.random.permutation(merged['test_x'].shape[0])
	np.take(merged['train_x'],shuffle_train,axis=0,out=merged['train_x'])
	np.take(merged['train_y'],shuffle_train,axis=0,out=merged['train_y'])
	np.take(merged['test_x'],shuffle_test,axis=0,out=merged['test_x'])
	np.take(merged['test_y'],shuffle_test,axis=0,out=merged['test_y'])
	return merged



def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, channels), dtype = np.float32)
            X_translated.fill(0.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr

def mnist_augmented_data():
	mnist = mnist_funct()
	translated_imgs = translate_images(mnist['train_x'])
	translated_imgs = np.concatenate([mnist['train_x'],translated_imgs],axis=0).reshape((-1,28,28,1))
	translated_labels = np.concatenate([mnist['train_y'] for _ in range(5)],axis=0)
	translated_labels = translated_labels.reshape((-1,))
	mnist['train_x'] = translated_imgs
	mnist['train_y'] = translated_labels
	return mnist
	





