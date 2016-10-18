#coding:utf-8

import mxnet as mx
import numpy as np
from PIL import Image, ImageEnhance
from collections import namedtuple
import math
import matplotlib.pyplot as plt
import os
import ImageFilter
import get_net

ctx = mx.gpu(0)
fcscnn_mod = namedtuple('fcscnn_mod',['exc', 'net', 'data_lcl', 'data_glb', 'arg_name_list', 'arg_dict', 'aux_dict'])

def get_model():
	fcscnn = get_net.get_net(False)
	arg_names = fcscnn.list_arguments()
	aux_names = fcscnn.list_auxiliary_states()
	arg_shapes, output_shapes, aux_shapes = fcscnn.infer_shape(data_lcl = (1, 3, 48, 48), data_glb = (1, 3, 48, 48))

#==================================================================================================

	arg_arrays = [mx.nd.zeros(shape) for shape in arg_shapes]
	arg_dict = dict(zip(arg_names, arg_arrays))
	aux_dict = {}
	for name, shape in zip(aux_names, aux_shapes):
		aux_dict[name] = mx.nd.zeros(shape)

#==================================================================================================

	exc = fcscnn.bind(ctx = mx.gpu(), args = arg_dict, aux_states = aux_dict)

	arg_name_list = []
	for name in arg_names:
		# print name
		if name in ['data_lcl', 'data_glb']:
			continue
		arg_name_list.append(name)

	data_lcl = exc.arg_dict['data_lcl']
	data_glb = exc.arg_dict['data_glb']

	return fcscnn_mod(exc = exc, net = fcscnn, data_lcl = data_lcl, data_glb = data_glb, arg_name_list = arg_name_list,
		arg_dict= arg_dict, aux_dict = aux_dict)

def test(raw_path, fix_path, fcscnn, k, i_idx, j_idx):
	arg_dict_params = mx.nd.load('.\\params\\saving\\arg_dict.params')
	aux_dict_params = mx.nd.load('.\\params\\saving\\aux_dict.params')

	for arg in fcscnn.arg_dict:
		for param in arg_dict_params:
			if arg == param:
				if fcscnn.arg_dict[arg].shape != arg_dict_params[param].shape:
					# print arg + ' is wrong'
					continue
				fcscnn.arg_dict[arg][:] = arg_dict_params[param].asnumpy()
				# print 'using old ' + arg

	for aux in fcscnn.aux_dict:
		for param in aux_dict_params:
			if aux == param:
				if fcscnn.aux_dict[aux].shape != aux_dict_params[param].shape:
					# print aux + ' is wrong'
					continue
				fcscnn.aux_dict[aux][:] = aux_dict_params[param].asnumpy()
				# print 'using old ' + aux

	# coord_list, idx_list = get_shuffled_coord_list()

	# raw = Image.fromarray(preprocessing_data(np.array(Image.open(raw_path)), 384).astype('uint8'))
	raw_0 = Image.open(testing_list[k])
	width_0 = np.array(raw_0).shape[1]
	height_0 = np.array(raw_0).shape[0]

	width_num_0 = width_0 / 48
	height_num_0 = height_0 / 48
	width_1 = width_num_0 * 48
	height_1 = height_num_0 * 48

	raw_1 = raw_0.crop((width_0/2-width_1/2, height_0/2-height_1/2, width_0/2+width_1/2, height_0/2+height_1/2))

	min_num = min(width_num_0, height_num_0)
	width_num = int(width_num_0 * (8/float(min_num)))
	height_num = int(height_num_0 * (8/float(min_num)))
	raw = raw_1.resize((width_num*48, height_num*48))
	width = width_num*48
	height = height_num*48
	# raw.save('.\\temp.jpg')

	glb = pad_the_raw(raw, width, height)
	# glb.save('.\\temp_2.jpg')
	salie_map = np.empty((width/6, height/6))
	for i in range(0, width_num):
		for j in range(0, height_num):

			region_loc_0 = np.array(raw.crop((i*48, j*48, (i+1)*48, (j+1)*48)).resize((48,48)))
			region_glb_0 = np.array(glb.crop((i*48, j*48, i*48+192, j*48+192)).resize((48,48)))

			fcscnn.data_lcl[:] = np.swapaxes(rgblize(region_loc_0, 48, 48), 0 ,2).reshape(1, 3, 48, 48)
			fcscnn.data_glb[:] = np.swapaxes(rgblize(region_glb_0, 48, 48), 0 ,2).reshape(1, 3, 48, 48)

			fcscnn.exc.forward(is_train = False)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

			current_img_output = fcscnn.exc.outputs[0].asnumpy()
			# recimg = current_img_output
			recimg = current_img_output.reshape((8, 8))

			salie_map[i*8:(i+1)*8, j*8:(j+1)*8] = np.array(recimg)

	print '======================================'
	print 'testing max: '+str(np.max(salie_map))
	print 'testing mea: '+str(np.mean(salie_map))
	print 'testing min: '+str(np.min(salie_map))
	print '======================================'

	img_name_predct = '.\\recons\\i' + str(k) + '_predct.jpeg'

	salie_full_size = np.zeros((width_0, height_0)) 

	# salie_map_0 = enhance(salie_map.transpose()) # (64L, 96L)
	# Image.fromarray(salie_map_0.astype('uint8')).save('.\\salie_map_0.jpg')
	salie_map_0 = salie_map.transpose()
	preact0 = Image.fromarray(salie_map_0.astype('uint8')).convert('L').filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur).resize((width_1, height_1))

	salie_full_size[width_0/2-width_1/2:width_0/2-width_1/2+width_1, height_0/2-height_1/2:height_0/2-height_1/2+height_1,] = np.array(preact0).transpose()
	Image.fromarray(salie_full_size.transpose().astype('uint8')).filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur).convert('L').filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur).save(img_name_predct, 'JPEG')

	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# def enhance(img_array_0):

# 	img_array_1 = img_array_0 * (255 / (np.max(img_array_0) - np.min(img_array_0)))
# 	img_array_2 = 1 / (np.ones(img_array_1.shape) * (1 / 255.0) + np.exp( - 0.04 * img_array_1))
# 	img_array = img_array_2 * (250 / (np.max(img_array_2) - np.min(img_array_2)))

# 	return img_array

def pad_the_raw(raw_img, width, height):
	glb = np.zeros((width+144, height+144 ,3))
	raw = np.array(raw_img)
	glb[72:width+72,72:height+72, :] = np.swapaxes(raw, 0, 1)
	glb[0:72,72:height+72, :] = raw[:,0, :].reshape(1,height,3)
	glb[72:width+72,0:72, :] = raw[0,:, :].reshape(width,1,3)
	glb[72:width+72,height+72:height+144, :] = raw[height-1,:, :].reshape(width,1,3)
	glb[width+72:width+144,72:height+72, :] = raw[:,width-1, :].reshape(1,height,3)
	glb[0:72,0:72,:] = glb[0:72,72,:]
	glb[width+72:width+144,0:72,:] = glb[width+72:width+144,72,:]
	glb[0:72,height+72:height+144,:] = glb[0:72:,height+71,:]
	glb[width+72:width+144,height+72:height+144,:] = glb[width+72:width+144,height+71,:]
	glb_img = Image.fromarray(np.swapaxes(glb,0,1).astype('uint8'))
	return glb_img

def preprocessing_data(data, width, height):
	data = rgblize(data, width, height)
	# std = np.std(data)
	# if std == 0:
	# 	data = data - np.mean(data)
	# else:
	# 	data = data - np.mean(data)
	# 	data = data / float(np.std(data))	
	return data

def get_local_regions(fix_path, raw_path):
	map_list = []
	raw_list = []
	fix = Image.open(fix_path)
	raw = Image.open(raw_path)

	for i in range(0,8):
		for j in range(0,8):

			fixx = fix.crop((i*48, j*48, (i+1)*48, (j+1)*48)).resize((8,8))
			map_list.append(fixx)
			raw_list.append(np.array(raw.crop((i*48, j*48, (i+1)*48, (j+1)*48))))

	return map_list, raw_list

def rgblize(region, width, height):
	if region.ndim == 3:
		# print region.shape
		return region
	elif region.ndim == 2:
		rgb_region = np.empty((3,width,height))
		rgb_region[0] = region.transpose()
		rgb_region[1] = region.transpose()
		rgb_region[2] = region.transpose()
		rgb_region = np.swapaxes(rgb_region, 0, 2)
		# print rgb_region.shape
		return rgb_region

def get_shuffled_coord_list():
	coords = []
	for i in range(0,8):
		for j in range(0,8):
			coords.append((j,i))
	idx_list = np.arange(64)
	np.random.shuffle(idx_list)
	shuffled_coords = []
	for i in range(0,len(coords)):
		shuffled_coords.append(coords[idx_list[i]])
	return shuffled_coords, idx_list

def vis_net():
	net = get_net.get_net(True)
	a = mx.viz.plot_network(net, shape={"data_lcl":(1, 1, 48, 48), "data_glb":(1, 1, 48, 48)}, node_attrs={"shape":'rect',"fixedsize":'false'})
	a.render("net_20161012")

# vis_net()
fcscnn = get_model()
test('.\\raw_000.jpeg', '.\\fix_000.jpeg', fcscnn, 0, 20000, 20000)

# img = np.array(Image.open('.\\100.jpeg'))

# width = int(img.shape[1] / 48) * 48
# height = int(img.shape[0] / 48) * 48
# img_raw = Image.open('.\\100.jpeg').resize((width, height))
# pad_the_raw(img_raw, width, height).save('.\\101.jpeg')