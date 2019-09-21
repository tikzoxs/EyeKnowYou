from __future__ import absolute_import, division, print_function

import tensorflow as tf 
import numpy as np 
import h5py
import train_generator as geny_tr
import cv2

datapath ="/media/tkal976/Transcend/Tharindu/new_eye_data/7_1.h5"
g = geny_tr.train_generator(16)
ds = tf.data.Dataset.from_generator(g, output_types=((tf.uint8, tf.uint8)))
print("printing DS")
print(ds)

value = ds.make_one_shot_iterator().get_next()

sess = tf.Session()

# Example on how to read elements
while True:
	try:
		print("trying")
		data = sess.run(value)
		print(data[0].shape)
		print(data[1])
		xdset = data[0]
		for j in range(64):
			cv2.imshow("eye", xdset[0,:,:,j,0])
			j = cv2.waitKey(2)
			if(j == 27):
				break
		if(j == 27):
				break
	except tf.errors.OutOfRangeError:
		print('done.')
		break