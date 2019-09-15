import cv2
import numpy as np
import h5py
import os

BLUR_THRESH = 10
BLUR_CONTINUITY = 6
blur_count = 0
blur_flag = False
global_image_count = 0
break_flag = False

def add_to_file(datapath, filepath, user, test):
	with h5py.File(datapath, mode='r') as h5f:
		X = h5f['X']
		Y = h5f['Y']
		# U = h5f['U']
		# T = h5f['T']
		h5_append(filepath, X, Y, test)
	

def h5_create(filepath):
	x_shape = (128, 192, 64)
	y_shape = (1,7)
	u_shape = (4,1) #user no, gender, age, eye color
	t_shape = (1,1)
	with h5py.File(filepath, mode='a') as h5f:
		xdset = h5f.create_dataset('X', (0,) + x_shape, maxshape=(None,) + x_shape, dtype='uint8', chunks=(100,) + x_shape)
		ydset = h5f.create_dataset('Y', (0,) + y_shape, maxshape=(None,) + y_shape, dtype='uint8', chunks=(100,) + y_shape)
		# udset = h5f.create_dataset('U', (0,) + u_shape, maxshape=(None,) + u_shape, dtype='uint8', chunks=(100,) + u_shape)
		# tdset = h5f.create_dataset('T', (0,) + t_shape, maxshape=(None,) + u_shape, dtype='uint8', chunks=(100,) + u_shape)


def resize_img(image):
	global break_flag
	data = np.zeros((128,192,64), dtype = np.uint8)
	dim = (192, 128)
	for i in range(64):
		small = cv2.resize(image[:,:,i], dim)
		data[:,:,i] = small
	return data
	

def h5_append(filepath, X, Y, test):
	print(test)
	global break_flag
	global global_image_count
	print("***** New File *****")
	with h5py.File(filepath, mode='a') as h5f:
		xdset = h5f['X']
		ydset = h5f['Y']
		# udset = h5f['U']
		# tdset = h5f['T']
		print(X.shape)
		print(Y.shape)
		# print(T.shape)
		# print(U.shape)
		i = 20
		while(i < int(X.shape[0]) - 20):
			data = resize_img(X[i])
			# if(break_flag):
			# 	break
			# # for j in range(64):
			# # 	cv2.imshow('image', data[:,:,j])
			# # 	j = cv2.waitKey(2)
			# # 	if(j == 27):
			# # 		break_flag = True
			# # 		break
			global_image_count = global_image_count + 1
			xdset.resize(xdset.shape[0]+1, axis=0)
			xdset[-1:] = data
			# print(xdset.shape)
			ydset.resize(ydset.shape[0]+1, axis=0)
			if(test == '0'):
				ydset[-1:] = np.transpose([0, 0, 0, 1, 1, 0, 0])
			elif(test == '1'):
				ydset[-1:] = np.transpose([0, 1, 0, 0, 1, 0, 0])
			elif(test == '2'):
				ydset[-1:] = np.transpose([0, 1, 0, 0, 0, 1, 0])
			elif(test == '3'):
				ydset[-1:] = np.transpose([0, 1, 0, 0, 0, 0, 1])
			elif(test == '4'):
				ydset[-1:] = np.transpose([1, 0, 0, 0, 1, 0, 0])
			elif(test == '5'):
				ydset[-1:] = np.transpose([1, 0, 0, 0, 0, 1, 0])
			elif(test == '6'):
				ydset[-1:] = np.transpose([1, 0, 0, 0, 0, 1, 0])
			elif(test == '7'):
				ydset[-1:] = np.transpose([1, 0, 0, 0, 0, 0, 1])
			elif(test == '8'):
				ydset[-1:] = np.transpose([0, 0, 1, 0, 1, 0, 0])
			elif(test == '9'):
				ydset[-1:] = np.transpose([0, 0, 1, 0, 0, 0, 1])
			else:
				print("No such test")
			# print(ydset.shape)

			# udset.resize(udset.shape[0]+1, axis=0)
			# udset[-1:] = U[0]
			# # print(udset.shape)

			# tdset.resize(tdset.shape[0]+1, axis=0)
			# tdset[-1:] = T[0]
			# # print(tdset.shape)
			i = i+1

def main():
	global break_flag
	# filepath = "/media/tkal976/Transcend/Tharindu/aeye_data/test.h5"
	# data_files_directory = "/media/tkal976/Transcend/Tharindu/new_eye_data/validation"
	filepath = "/media/tkal976/Transcend/Tharindu/EyeKnowYou_data/train/train.h5"
	data_files_directory = "/media/tkal976/Transcend/Tharindu/new_eye_2/train"
	# filepath = "/home/tkal976/Desktop/Black/Codes/h5edit/temp/Aeye.h5"
	# data_files_directory = "/home/tkal976/Desktop/Black/Codes/h5edit/edit"
	h5_create(filepath)
	for datapath in os.listdir(data_files_directory):
		# if(datapath != 'test' and datapath != 'validation' and datapath != 'one' and datapath != 'one_person'):
		print(datapath)
		print('***** global_image_count = ' + str(global_image_count) + " *****")
		if(break_flag):
			break
		user = datapath.split("_")[0]
		test = datapath.split("_")[1].split('.')[0]
		add_to_file(data_files_directory + '/' + datapath, filepath, user, test)

main()