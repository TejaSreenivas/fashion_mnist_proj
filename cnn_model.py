import tensorflow as tf
import numpy as np
import os 

def create_conv_layer(inp,filters,stride,padding):
	wt = tf.get_variable(name='weight',initializer=tf.truncated_normal(shape=filters,mean=0.0,stddev=0.1))
	conv = tf.nn.conv2d(input=inp,filter=wt,strides=stride,padding="SAME")
	b = tf.get_variable(name='bias',initializer=tf.truncated_normal(shape=[filters[-1]],mean=0.0,stddev=0.1))
	out = tf.add(conv,b)
	act = tf.nn.relu(out)
	return act

def create_dense_layer(inp,input_size,output_size):
	return tf.layers.dense(inp,units = output_size)
class model:
	"""
	param:
	param[0]  -  channel
	param[1]  -  filter1_size,# filters
	param[2]  -  filter2_size,# filters
	param[3]  -  dense size
	param[4]  -  output_size
	"""
	def cnn_2d(self,data,is_train,prob_keep,param):
		with tf.variable_scope("conv_1",reuse = tf.AUTO_REUSE):
			#data = tf.layers.batch_normalization(data,training=is_train)
			conv1 = create_conv_layer(data,[ param[1][0], param[1][0], param[0], param[1][1] ], [1]*4, "SAME")
			#conv1 = tf.layers.dropout(conv1,rate=0.4,training=is_train)
			conv1 = tf.layers.max_pooling2d(conv1,strides=2,padding='same')

		with tf.variable_scope("conv_2",reuse = tf.AUTO_REUSE):
			conv2 = create_conv_layer(conv1, [ param[2][0], param[2][0], param[1][1], param[2][1] ], [1]*4, "SAME")
			#conv2 = tf.layers.dropout(conv2,rate=0.4,training=is_train)
			conv2 = tf.nn.max_pooling2d(conv2,strides=2, padding = 'same')
			#conv2 = tf.layers.batch_normalization(conv2,training=is_train)

		with tf.variable_scope("flatten",reuse = tf.AUTO_REUSE):
			flat = tf.reshape(conv2, [ -1,7*7*param[2][1] ] )

		h1 = create_dense_layer(flat, input_size = 7*7*param[2][1], output_size = param[3])
		h1 = tf.nn.relu(h1)
		h1 = tf.layers.dropout(h1,rate = prob_keep, training = is_train)

		#logits
		y = create_dense_layer( h1, input_size = 7*7*param[2][1], output_size = param[4] )
		return y

	"""
	param:
	param[0]  -  channel
	param[1]  -  filter1_size,# filters
	param[2]  -  filter2_size,# filters
	param[3]  -  filter3_size,# filters
	param[4]  -  dense size
	param[5]  -  output_size
	"""
	def cnn_2d_3l(self,data,is_train,prob_keep,param):
		with tf.variable_scope("conv_1",reuse = tf.AUTO_REUSE):
			conv1 = create_conv_layer(data,[ param[1][0], param[1][0], param[0], param[1][1] ], [1]*4, "SAME")
			conv1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')

		with tf.variable_scope("conv_2",reuse = tf.AUTO_REUSE):
			conv2 = create_conv_layer(conv1, [ param[2][0], param[2][0], param[1][1], param[2][1] ], [1]*4, "SAME")
			conv2 = tf.nn.max_pool(conv2,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')

		with tf.variable_scope("conv_3",reuse = tf.AUTO_REUSE):
			conv3 = create_conv_layer(conv2, [ param[3][0], param[3][0], param[2][1], param[3][1] ], [1]*4, "SAME")
			conv3 = tf.nn.max_pool(conv3,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')
			conv3 = tf.layers.batch_normalization(conv3,training=is_train)


		with tf.variable_scope("flatten",reuse = tf.AUTO_REUSE):
			flat = tf.reshape(conv3, [ -1,4*4*param[3][1] ] )

		h1 = create_dense_layer(flat, input_size = 7*7*param[2][1], output_size = param[3])
		h1 = tf.nn.relu(h1)
		h1 = tf.layers.dropout(h1,rate = prob_keep, training = is_train)
		#logits
		y = create_dense_layer( h1, input_size = param[3], output_size = param[4] )
		return y


	def save_weights(self):
		# yet to be written 
		pass

	def total_params(self):
		count = 0
		info = ""
		for var in tf.trainable_variables():
			shape = var.get_shape()
			p = 1
			info = info + "varname : "+var.name+"-("
			print("varname : " + var.name,end=' ')
			l = []
			for d in shape:
				p*=d
				info = info + " " + str(d)
				l.append(d)
			count+=p
			print(l)
			info = info+" )$"
		info = info + "total param count : "+str(count)
		print("total number of trainable parameter : {}".format(count))
		return info
		

	def __init__(self, data,is_train,prob_keep,param):
		
		if len(param) == 5:
			self.logits = self.cnn_2d(data,is_train,prob_keep,param)
		elif len(param) == 6:
			self.logits = self.cnn_2d_3l(data,is_train,prob_keep,param)
		#end
		