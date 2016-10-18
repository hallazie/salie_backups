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

def test(raw_path, fix_path, fcscnn, k, i, j_idx):
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

	coord_list, idx_list = get_shuffled_coord_list()

	# raw = Image.fromarray(preprocessing_data(np.array(Image.open(raw_path)), 384).astype('uint8'))
	raw = Image.open(raw_path)
	glb = pad_the_raw(raw)

	salie_map = np.empty((64,64))
	for j in range(0, 64):

		region_idx = idx_list[j]
		x = coord_list[j][0]
		y = coord_list[j][1]

		region_loc_0 = np.array(raw.crop((x*48, y*48, (x+1)*48, (y+1)*48)).resize((48,48)))
		region_glb_0 = np.array(glb.crop((x*48, y*48, x*48+192, y*48+192)).resize((48,48)))
		# region_loc = region_loc_0
		# region_glb = preprocessing_data(region_glb_0, 48)

		fcscnn.data_lcl[:] = np.swapaxes(rgblize(region_loc_0, 48), 0 ,2).reshape(1, 3, 48, 48)
		fcscnn.data_glb[:] = np.swapaxes(rgblize(region_glb_0, 48), 0 ,2).reshape(1, 3, 48, 48)

		fcscnn.exc.forward(is_train = False)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		m = region_idx % 8
		n = region_idx / 8
		current_img_output = fcscnn.exc.outputs[0].asnumpy()
		recimg = current_img_output.transpose()
		recimg = recimg.reshape((8, 8))

		salie_map[n*8:(n+1)*8, m*8:(m+1)*8] = np.array(recimg)

	print '======================================'
	print 'testing max: '+str(np.max(salie_map))
	print 'testing mea: '+str(np.mean(salie_map))
	print 'testing min: '+str(np.min(salie_map))
	print '======================================'
	img_name_predct = '.\\recons\\' + str(k) + '_' + str(i) + '_' + str(j_idx) + '_predct.jpeg'
	img_name_rawimg = '.\\recons\\' + str(k) + '_' + str(i) + '_' + str(j_idx) + '_rawimg.jpeg'
	img_name_fiximg = '.\\recons\\' + str(k) + '_' + str(i) + '_' + str(j_idx) + '_fiximg.jpeg'

	# salie_map = salie_map * (255.0 / float(np.max(salie_map) - np.min(salie_map)))

	preact0 = Image.fromarray(salie_map).convert('L').filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur)
	preact = ImageEnhance.Contrast(preact0).enhance(2.0) # actural  brightness
	Image.open(raw_path).save(img_name_rawimg, 'JPEG')
	Image.open(fix_path).save(img_name_fiximg, 'JPEG')
	preact.resize((384,384)).filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur).convert('L').save(img_name_predct, 'JPEG')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def pad_the_raw(raw_img):
	glb = np.zeros((528, 528 ,3))
	raw = rgblize(np.array(raw_img), 384)
	glb[72:456,72:456, :] = raw
	glb[0:72,72:456, :] = raw[0,:, :].reshape(1,384,3)
	glb[72:456,0:72, :] = raw[:,0, :].reshape(384,1,3)
	glb[72:456,456:528, :] = raw[:,383, :].reshape(384,1,3)
	glb[456:528,72:456, :] = raw[383,:, :].reshape(1,384,3)
	glb[0:72,0:72,:] = glb[0:72,72,:]
	glb[456:584,0:72,:] = glb[456:584,72,:]
	glb[0:72,456:584,:] = glb[0:72:,455,:]
	glb[456:584,456:584,:] = glb[456:584,455,:]
	glb_img = Image.fromarray(glb.astype('uint8'))
	return glb_img

def preprocessing_data(data, size):
	data = rgblize(data, size)
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

def rgblize(region, size):
	if region.ndim == 3:
		# print region.shape
		return region
	elif region.ndim == 2:
		rgb_region = np.empty((3,size,size))
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
	net = get_net()
	a = mx.viz.plot_network(net, shape={"data_lcl":(1, 1, 48, 48), "data_glb":(1, 1, 48, 48)}, node_attrs={"shape":'rect',"fixedsize":'false'})
	a.render("laz_salie_2")

# vis_net()

# test('.\\dataset\\raws\\raw_7.jpg', '.\\dataset\\fixs\\fix_7.jpg', 20000, 20000)