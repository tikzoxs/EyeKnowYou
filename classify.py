import argparse
import os

import tensorflow as tf
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.utils import np_utils

import classify_generator as geny

LAYER_ACTIVATION = 'relu'

def main():
    global batch_size
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--filepath', required=True, help='directions to folder containing H5 files to be classified')
    args = parser.parse_args()

    nb_classes = args.nclass

    # Define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 15), input_shape=(128, 192, 64, 1), border_mode='same', name='new_input'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same')) #, trainable = False
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid', name='new_dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', name='new_output'))

    result_direct = "/home/tkal976/Desktop/Black/Codes/git/3DCNN/result_dir/"
    model.load_weights(result_direct + "eky_weights.hd5", by_name=True)
    

    batch_size = args.batch
    filepath = args.filepath

    test_gen = geny.classify_generator(batch_size)

    predictions = model.predict(test_gen.__next__())


if __name__ == '__main__':
    main()