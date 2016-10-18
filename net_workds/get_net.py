#coding:utf-8

import mxnet as mx
import numpy as np

def get_net(is_train):
	data_glb = mx.symbol.Variable('data_glb')
	data_lcl = mx.symbol.Variable('data_lcl')
	label = mx.symbol.Variable('label')
	# merge = mx.symbol.Variable('merge')
	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# local
	conv_lcl_0_weight = mx.symbol.Variable('conv_lcl_0_weight') # vgg conv1_1
	conv_lcl_1_weight = mx.symbol.Variable('conv_lcl_1_weight') # vgg conv1_2
	conv_lcl_2_weight = mx.symbol.Variable('conv_lcl_2_weight') # vgg conv2_1
	conv_lcl_3_weight = mx.symbol.Variable('conv_lcl_3_weight') # vgg conv2_2
	conv_lcl_4_weight = mx.symbol.Variable('conv_lcl_4_weight') # vgg conv3_1
	conv_lcl_5_weight = mx.symbol.Variable('conv_lcl_5_weight') # vgg conv3_2
	conv_lcl_6_weight = mx.symbol.Variable('conv_lcl_6_weight') # vgg conv3_3
	conv_lcl_7_weight = mx.symbol.Variable('conv_lcl_7_weight') # vgg conv3_4
	conv_glb_0_weight = mx.symbol.Variable('conv_glb_0_weight') # vgg conv1_1
	conv_glb_1_weight = mx.symbol.Variable('conv_glb_1_weight') # vgg conv1_2
	# merge = mx.symbol.Variable('merge')

	# local struct
	conv_lcl_0 = mx.symbol.Convolution(data = data_lcl, name = 'conv_lcl_0', num_filter = 64, kernel = (3,3), weight = conv_lcl_0_weight) # 64*46*46
	bn_lcl_0 = mx.symbol.BatchNorm(data = conv_lcl_0, name = 'bn_lcl_0')
	act_lcl_0 = mx.symbol.Activation(data = bn_lcl_0, name = 'act_lcl_0', act_type = 'relu')

	conv_lcl_1 = mx.symbol.Convolution(data = act_lcl_0, name = 'conv_lcl_1', num_filter = 64, kernel = (3,3), weight = conv_lcl_1_weight) # 64*44*44
	bn_lcl_1 = mx.symbol.BatchNorm(data = conv_lcl_1, name = 'bn_lcl_1')
	act_lcl_1 = mx.symbol.Activation(data = bn_lcl_1, name = 'act_lcl_1', act_type = 'relu')

	conv_lcl_2 = mx.symbol.Convolution(data = act_lcl_1, name = 'conv_lcl_2', num_filter = 128, kernel = (3,3), weight = conv_lcl_2_weight) # 128*42*42
	bn_lcl_2 = mx.symbol.BatchNorm(data = conv_lcl_2, name = 'bn_lcl_2')
	act_lcl_2 = mx.symbol.Activation(data = bn_lcl_2, name = 'act_lcl_2', act_type = 'relu')

	conv_lcl_3 = mx.symbol.Convolution(data = act_lcl_2, name = 'conv_lcl_3', num_filter = 128, kernel = (3,3), weight = conv_lcl_3_weight) # 128*40*40
	bn_lcl_3 = mx.symbol.BatchNorm(data = conv_lcl_3, name = 'bn_lcl_3')
	act_lcl_3 = mx.symbol.Activation(data = bn_lcl_3, name = 'act_lcl_3', act_type = 'relu')
	pool_lcl_3 = mx.symbol.Pooling(data = act_lcl_3, name = 'pool_lcl_3', stride = (2,2), kernel = (2,2), pool_type = 'max') # 128*20*20

	# conv_lcl_3_h = mx.symbol.Convolution(data = pool_lcl_3, name = 'conv_lcl_3_h', num_filter = 128, kernel = (3,3), dilate = (2,2)) # 128*40*40
	# bn_lcl_3_h = mx.symbol.BatchNorm(data = conv_lcl_3_h, name = 'bn_lcl_3_h')
	# act_lcl_3_h = mx.symbol.Activation(data = bn_lcl_3_h, name = 'act_lcl_3_h', act_type = 'relu')

	conv_lcl_4 = mx.symbol.Convolution(data = pool_lcl_3, name = 'conv_lcl_4', num_filter = 256, kernel = (3,3), weight = conv_lcl_4_weight) # 256*18*18
	bn_lcl_4 = mx.symbol.BatchNorm(data = conv_lcl_4, name = 'bn_lcl_4')
	act_lcl_4 = mx.symbol.Activation(data = bn_lcl_4, name = 'act_lcl_4', act_type = 'relu')

	conv_lcl_5 = mx.symbol.Convolution(data = act_lcl_4, name = 'conv_lcl_5', num_filter = 256, kernel = (3,3), weight = conv_lcl_5_weight) # 256*16*16
	bn_lcl_5 = mx.symbol.BatchNorm(data = conv_lcl_5, name = 'bn_lcl_5')
	act_lcl_5 = mx.symbol.Activation(data = bn_lcl_5, name = 'act_lcl_5', act_type = 'relu')

	conv_lcl_6 = mx.symbol.Convolution(data = act_lcl_5, name = 'conv_lcl_6', num_filter = 256, kernel = (3,3), weight = conv_lcl_6_weight, dilate = (2,2)) # 256*12*12
	bn_lcl_6 = mx.symbol.BatchNorm(data = conv_lcl_6, name = 'bn_lcl_6')
	act_lcl_6 = mx.symbol.Activation(data = bn_lcl_6, name = 'act_lcl_6', act_type = 'relu')

	conv_lcl_7 = mx.symbol.Convolution(data = act_lcl_6, name = 'conv_lcl_7', num_filter = 256, kernel = (3,3), weight = conv_lcl_7_weight, dilate = (2,2)) # 256*8*8
	bn_lcl_7 = mx.symbol.BatchNorm(data = conv_lcl_7, name = 'bn_lcl_7')
	act_lcl_7 = mx.symbol.Activation(data = bn_lcl_7, name = 'act_lcl_7', act_type = 'relu')

	conv_lcl_8 = mx.symbol.Convolution(data = act_lcl_7, name = 'conv_lcl_8', num_filter = 1, kernel = (1,1))

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# # global
	# back up glb struct
	# holes..................
	conv_glb_0 = mx.symbol.Convolution(data = data_glb, name = 'conv_glb_0', num_filter = 64, kernel = (3,3), weight = conv_glb_0_weight) # 32*46*46
	bn_glb_0 = mx.symbol.BatchNorm(data = conv_glb_0, name = 'bn_glb_0')
	act_glb_0 = mx.symbol.Activation(data = bn_glb_0, name = 'act_glb_0', act_type = 'relu')


	conv_glb_1 = mx.symbol.Convolution(data = act_glb_0, name = 'conv_glb_1', num_filter = 64, kernel = (3,3), weight = conv_glb_1_weight) # 32*44*44
	bn_glb_1 = mx.symbol.BatchNorm(data = conv_glb_1, name = 'bn_glb_1')
	act_glb_1 = mx.symbol.Activation(data = bn_glb_1, name = 'act_glb_1', act_type = 'relu')
	
	# merge_0 = act_lcl_1 + act_glb_1
	conv_glb_2 = mx.symbol.Convolution(data = act_glb_1, name = 'conv_glb_2', num_filter = 32, kernel = (3,3), dilate = (2,2)) # 32*40*40
	bn_glb_2 = mx.symbol.BatchNorm(data = conv_glb_2, name = 'bn_glb_2')
	act_glb_2 = mx.symbol.Activation(data = bn_glb_2, name = 'act_glb_2', act_type = 'relu')
	pool_glb_2 = mx.symbol.Pooling(data = act_glb_2, name = 'pool_glb_2', stride = (2,2), kernel = (2,2), pool_type = 'max')
	
	conv_glb_3 = mx.symbol.Convolution(data = pool_glb_2, name = 'conv_glb_3', num_filter = 32, kernel = (3,3), dilate = (2,2)) # 32*16*16
	bn_glb_3 = mx.symbol.BatchNorm(data = conv_glb_3, name = 'bn_glb_3')
	act_glb_3 = mx.symbol.Activation(data = bn_glb_3, name = 'act_glb_3', act_type = 'relu')
	# drop_glb_4 = mx.symbol.Dropout(data = act_glb_3)
	
	conv_glb_4 = mx.symbol.Convolution(data = act_glb_3, name = 'conv_glb_4', num_filter = 32, kernel = (3,3), dilate = (2,2)) # 32*12*12
	bn_glb_4 = mx.symbol.BatchNorm(data = conv_glb_4, name = 'bn_glb_4')
	act_glb_4 = mx.symbol.Activation(data = bn_glb_4, name = 'act_glb_4', act_type = 'relu')

	conv_glb_5 = mx.symbol.Convolution(data = act_glb_4, name = 'conv_glb_5', num_filter = 1, kernel = (3,3), dilate = (2,2)) # 32*8*8
	# bn_glb_5 = mx.symbol.BatchNorm(data = conv_glb_5, name = 'bn_glb_5')
	# act_glb_5 = mx.symbol.Activation(data = bn_glb_5, name = 'act_glb_5', act_type = 'relu')

	# conv_glb_6 = mx.symbol.Convolution(data = act_glb_5, name = 'conv_glb_6', num_filter = 1, kernel = (1,1)) # 1*8*8
	# bn_glb_6 = mx.symbol.BatchNorm(data = conv_glb_6, name = 'bn_glb_6')
	# act_glb_6 = mx.symbol.Activation(data = bn_glb_6, name = 'act_glb_6', act_type = 'relu')
	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# merge
	merge = conv_lcl_8 + conv_glb_5

	final = modified_sigmoid(merge)

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# out = mx.symbol.MAERegressionOutput(data = conv_3, label = label, name = 'out')
	out = mx.symbol.LinearRegressionOutput(data = final, label = label, name = 'out')
	# out = mx.symbol.LogisticRegressionOutput(data = conv_4, label = label, name = 'out')

	if is_train == True:
		return out
	else:
		return merge

def modified_sigmoid(merge):
	return 1 / ((1 / 255.0) + mx.symbol.exp( - 0.03 * merge))

# def vis_net():
# 	net = get_net(True)
# 	a = mx.viz.plot_network(net, shape={"data_lcl":(1, 1, 48, 48), "data_glb":(1, 1, 48, 48)}, node_attrs={"shape":'rect',"fixedsize":'false'})
# 	a.render("laz_salie_3")

# vis_net()