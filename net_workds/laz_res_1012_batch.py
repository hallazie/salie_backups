#coding:utf-8

import mxnet as mx
import numpy as np
from PIL import Image, ImageEnhance
from collections import namedtuple
import math
import matplotlib.pyplot as plt
import os
import ImageFilter
from datetime import timedelta, datetime
import get_net
import test

ctx = mx.gpu(0)
fcscnn_mod = namedtuple('fcscnn_mod',['exc', 'net', 'data_lcl', 'data_glb', 'label','arg_name_list', 'arg_dict', 'grd_dict', 'aux_dict'])
num_epoch = 5

def get_model():
	fcscnn = get_net.get_net(True)
	arg_names = fcscnn.list_arguments()
	aux_names = fcscnn.list_auxiliary_states()
	arg_shapes, output_shapes, aux_shapes = fcscnn.infer_shape(data_lcl = (128, 3, 48, 48), data_glb = (128, 3, 48, 48))
	initializer = mx.initializer.Normal(0.01)

	for name,shape in zip(arg_names, arg_shapes):
		print name + ':' + str(shape)
	print 'output:' + str(output_shapes)

#==================================================================================================

	arg_arrays = [mx.nd.zeros(shape) for shape in arg_shapes]

	arg_dict = dict(zip(arg_names, arg_arrays))
	grd_dict = {}
	for name, shape in zip(arg_names, arg_shapes):
		if name in ['data_lcl', 'data_glb', 'label']:
			continue
		grd_dict[name] = mx.nd.zeros(shape)
		# print 'init..' + name + ' in shape of..' + str(shape)
		initializer(name, arg_dict[name])
	aux_dict = {}
	for name, shape in zip(aux_names, aux_shapes):
		aux_dict[name] = mx.nd.zeros(shape)
#==================================================================================================

	exc = fcscnn.bind(ctx = mx.gpu(0), args = arg_dict, args_grad = grd_dict, aux_states = aux_dict, grad_req = 'add')

	arg_name_list = []
	for name in arg_names:
		if name in ['data_lcl', 'data_glb', 'label']:
			continue
		arg_name_list.append(name)

	data_lcl = exc.arg_dict['data_lcl']
	data_glb = exc.arg_dict['data_glb']
	label = exc.arg_dict['label']

	return fcscnn_mod(exc = exc, net = fcscnn, data_lcl = data_lcl, data_glb = data_glb, label = label, arg_name_list = arg_name_list,
		arg_dict= arg_dict, grd_dict = grd_dict, aux_dict = aux_dict)

def train():
	fcscnn = get_model()
	fcscnn_test = test.get_model()
	raw_list_temp, fix_list_temp = get_all_training_list()
	learning_rate = 0.002
	momentum = 0.9
	max_grad_norm = 5000
	# optmzr = mx.optimizer.SGD(momentum = momentum, learning_rate = learning_rate, wd = 0.0002)
	optmzr = mx.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-07, decay_factor=0.99999999)
	# optmzr = mx.optimizer.RMSProp(gamma1=0.95, gamma2=0.9)
	# optmzr = mx.optimizer.SGLD(learning_rate = learning_rate, wd = 0.0002, rescale_grad = 1.0, clip_gradient = 10.0)
	# optmzr = mx.optimizer.AdaDelta(rho=0.9, epsilon=1e-05, wd = 0.0002)
	# optmzr = mx.optimizer.AdaGrad(eps=1e-07)
	# optmzr = mx.optimizer.ccSGD(momentum=0.9, rescale_grad=1.0, clip_gradient=-1.0)

	updater = mx.optimizer.get_updater(optmzr)

	arg_dict_params = mx.nd.load('.\\params\\saving\\arg_dict.params')
	aux_dict_params = mx.nd.load('.\\params\\saving\\aux_dict.params')

	for arg in fcscnn.arg_dict:
		for param in arg_dict_params:
			if arg == param:
				if fcscnn.arg_dict[arg].shape != arg_dict_params[param].shape:
					print arg + ' is wrong'
					continue
				fcscnn.arg_dict[arg][:] = arg_dict_params[param].asnumpy()
				print 'using old ' + arg

	for aux in fcscnn.aux_dict:
		for param in aux_dict_params:
			if aux == param:
				if fcscnn.aux_dict[aux].shape != aux_dict_params[param].shape:
					print aux + ' is wrong'
					continue
				fcscnn.aux_dict[aux][:] = aux_dict_params[param].asnumpy()
				print 'using old ' + aux

	# params = mx.nd.load('.\\params\\vgg19.params')
	# fcscnn.arg_dict['conv_lcl_0_weight'][:] = params['arg:conv1_1_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_1_weight'][:] = params['arg:conv1_2_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_2_weight'][:] = params['arg:conv2_1_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_3_weight'][:] = params['arg:conv2_2_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_4_weight'][:] = params['arg:conv3_1_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_5_weight'][:] = params['arg:conv3_2_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_6_weight'][:] = params['arg:conv3_3_weight'].asnumpy()
	# fcscnn.arg_dict['conv_lcl_7_weight'][:] = params['arg:conv3_4_weight'].asnumpy()
	# fcscnn.arg_dict['conv_glb_0_weight'][:] = params['arg:conv1_1_weight'].asnumpy()
	# fcscnn.arg_dict['conv_glb_1_weight'][:] = params['arg:conv1_2_weight'].asnumpy()

	for k in range(0, num_epoch):
		raw_list, fix_list, _ = shuffle_list(raw_list_temp, fix_list_temp)
		for i in range(0,int(1980/40)):
			raw_list_patch = raw_list[i*40 : (i+1)*40] # 40张图, 40*32, 共10个minibatch，每个minibatch含有128个regions
			fix_list_patch = fix_list[i*40 : (i+1)*40] # 40张fix map
			# lcl_region_list has 40*32=1280 regions
			lcl_region_list, glb_region_list, fix_region_list  = get_shuffled_regions(raw_list_patch, fix_list_patch)
			
			print 'this batch has ' + str(len(lcl_region_list)) + ' patches'
			if len(lcl_region_list) < 1280:
				print 'jump over this 10 batch for not enough valid patches...'
				continue

			for j in range(0,10):
				batch_lcl = np.empty((128, 3, 48, 48))
				batch_glb = np.empty((128, 3, 48, 48))
				batch_fix = np.empty((128, 1, 8, 8))
				for idx in range(0,128):
					batch_lcl[idx] = lcl_region_list[j*128+idx]
					batch_glb[idx] = glb_region_list[j*128+idx]
					batch_fix[idx] = fix_region_list[j*128+idx]

				# print np.max(batch_lcl)
				# print np.min(batch_lcl)
				# print np.max(batch_glb)
				# print np.min(batch_glb)
				fcscnn.data_lcl[:] = batch_lcl
				fcscnn.data_glb[:] = batch_glb
				fcscnn.label[:] = batch_fix
				fcscnn.exc.forward(is_train = True)
				fcscnn.exc.backward()

				print 'max: '+str(np.max(fcscnn.exc.outputs[0].asnumpy()))+'...'+str((np.max(batch_fix)))
				print 'mea: '+str(np.mean(fcscnn.exc.outputs[0].asnumpy()))+'...'+str((np.mean(batch_fix)))
				print 'min: '+str(np.min(fcscnn.exc.outputs[0].asnumpy()))+'...'+str((np.min(batch_fix)))

				norm = 0
				for name in fcscnn.grd_dict:
					l2_norm = mx.nd.norm(fcscnn.grd_dict[name] / 128.0).asscalar()
					norm = norm + l2_norm * l2_norm
				norm = math.sqrt(norm)

				print str(i) + 'th batch pack end at ' + str(i*40+j*4) + 'th pic with grad norm: ' + str(norm) + ' ,epoch ' + str(k) + ', at ' + str(datetime.utcnow())

				for idx in range(0, len(fcscnn.arg_name_list)):
					name = fcscnn.arg_name_list[idx]
					# fcscnn.grd_dict[name] \\= 64.0
					if norm > max_grad_norm:
						fcscnn.grd_dict[name][:] = fcscnn.grd_dict[name] * (max_grad_norm / norm)
					updater(index = idx, weight = fcscnn.arg_dict[name], grad = fcscnn.grd_dict[name])#, state = fcscnn.aux_dict[name])
					fcscnn.grd_dict[name][:] = 0
				# ================================================================================
				mx.nd.save('.\\params\\saving\\arg_dict.params', fcscnn.arg_dict)
				mx.nd.save('.\\params\\saving\\grd_dict.params', fcscnn.grd_dict)
				mx.nd.save('.\\params\\saving\\aux_dict.params', fcscnn.aux_dict)

				# test ...
				if j%2 == 0:
					rand_idx = np.random.randint(0, 1980)
					raw_img = raw_list[rand_idx]
					fix_img = fix_list[rand_idx]
					print 'testing on ' + str(i) + 'th img :' + raw_img
					test.test(raw_img, fix_img, fcscnn_test, k , i, j)

def get_shuffled_regions(raw_list, fix_list):
	lcl_region_list_0 = []
	glb_region_list_0 = []
	fix_region_list_0 = []
	for k in range(0,40):
		# raw_img = Image.fromarray(preprocessing_data(np.array(Image.open(raw_list[k])), 384).astype('uint8'))
		raw_img = Image.open(raw_list[k])
		fix_img = Image.open(fix_list[k]).convert('L')
		glb_img = pad_the_raw(raw_img)

		fixed_count = 0
		nonfixed_count = 0
		max_pix = np.max(np.array(fix_img))

		endtime1 = datetime.utcnow() + timedelta(seconds = 30)
		while fixed_count < 12:
			if datetime.utcnow() > endtime1:
				print 'not enough fixed...'
				return [], [], []
			x = np.random.randint(24, 384 - 24)
			y = np.random.randint(24, 384 - 24)
			if np.array(fix_img)[x][y] > max_pix*0.75 :
				lcl_0 = np.array(raw_img.crop((x-24, y-24, x+24, y+24)).resize((48,48)))
				lcl = np.swapaxes(rgblize(lcl_0, 48), 0, 2).reshape(1, 3, 48, 48)
				glb_0 = np.array(glb_img.crop((x-24, y-24, x+168, y+168)).resize((48,48)))
				glb = np.swapaxes(rgblize(glb_0, 48), 0, 2).reshape(1, 3, 48, 48)
				fix = np.array(fix_img.crop((x-24, y-24, x+24, y+24)).resize((8,8))).reshape(1 ,1, 8, 8)
				lcl_region_list_0.append(lcl)
				glb_region_list_0.append(glb)
				fix_region_list_0.append(fix)
				fixed_count += 1

		endtime2 = datetime.utcnow() + timedelta(seconds = 30)
		while nonfixed_count < 20:
			if datetime.utcnow() > endtime2:
				print 'not enough non-fixed...'
				return [], [], []
			x = np.random.randint(24, 384 - 24)
			y = np.random.randint(24, 384 - 24)
			if np.array(fix_img)[x][y] < max_pix*0.25 :
				lcl_0 = np.array(raw_img.crop((x-24, y-24, x+24, y+24)).resize((48,48)))
				lcl = np.swapaxes(rgblize(lcl_0, 48), 0, 2).reshape(1, 3, 48, 48)
				glb_0 = np.array(glb_img.crop((x-24, y-24, x+168, y+168)).resize((48,48)))
				glb = np.swapaxes(rgblize(glb_0, 48), 0, 2).reshape(1, 3, 48, 48)
				fix = np.array(fix_img.crop((x-24, y-24, x+24, y+24)).resize((8,8))).reshape(1 ,1, 8, 8)
				lcl_region_list_0.append(lcl)
				glb_region_list_0.append(glb)
				fix_region_list_0.append(fix)
				nonfixed_count += 1

	idx_list = np.arange(len(fix_region_list_0))
	np.random.shuffle(idx_list)

	lcl_region_list = shuffle_by_idx(idx_list, lcl_region_list_0)
	glb_region_list = shuffle_by_idx(idx_list, glb_region_list_0)
	fix_region_list = shuffle_by_idx(idx_list, fix_region_list_0)
	return lcl_region_list, glb_region_list, fix_region_list

def shuffle_by_idx(idx_list, target_list):
	shuffled_target_list = []
	for i in range(0, len(target_list)):
		shuffled_target_list.append(target_list[idx_list[i]])
	return shuffled_target_list

def preprocessing_data(data, size):
	data = rgblize(data, size)
	std = np.std(data)
	if std == 0:
		data = data - np.mean(data)
	else:
		data = data - np.mean(data)
		data = data / std
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
	# feaimg = plt.imshow(glb_img)
	# plt.savefig('.\\00000.jpg')
	# plt.close('all')
	return glb_img


def get_all_training_list():
	raw_list = []
	fix_list = []

	for i in range(0, 1980):
		raw_list.append('.\\cat2000\\raws\\raw_' + str(i) + '.jpg')
		fix_list.append('.\\cat2000\\fixs\\fix_' + str(i) + '.jpg')

	# for _, _, files in os.walk('.\\objects\\fix\\'):
	# 	for file in files:
	# 		fix_list.append('.\\objects\\fix\\' + file)

	# for _, _, files in os.walk('.\\objects\\raw\\'):
	# 	for file in files:
	# 		raw_list.append('.\\objects\\raw\\' + file)

	return raw_list, fix_list

def rgblize(region, size):
	if region.ndim == 3:
		# print region.shape
		# print 'rgb!'
		return region
	elif region.ndim == 2:
		# print 'gray!'
		rgb_region = np.empty((3,size,size))
		rgb_region[0] = region.transpose()
		rgb_region[1] = region.transpose()
		rgb_region[2] = region.transpose()
		rgb_region = np.swapaxes(rgb_region, 0, 2)
		# print rgb_region.shape
		return rgb_region

def shuffle_list(data_list, label_list):
	idx_list = np.arange(len(data_list))
	np.random.shuffle(idx_list)
	# print idx_list
	shuffled_data_list = []
	shuffled_label_list = []
	for i in range(0, len(data_list)):
		shuffled_data_list.append(data_list[idx_list[i]])
		shuffled_label_list.append(label_list[idx_list[i]])
	return shuffled_data_list, shuffled_label_list, idx_list

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
	a = mx.viz.plot_network(net, shape={"data_lcl":(128, 1, 48, 48), "data_glb":(128, 1, 48, 48)}, node_attrs={"shape":'rect',"fixedsize":'false'})
	a.render("laz_salie_batch")

if __name__ == '__main__':
	train()
	# vis_net()
	# aux_dict_params = mx.nd.load('.\\params\\20160924\\aux_dict.params')	
	# for param in aux_dict_params:
	# 	print param
	# image = Image.fromarray(rgblize(np.array(Image.open('.\\dataset\\stimuli\\raw_11.jpg')), 384).astype('uint8'))
	# image.save('.\\rgb.jpg')