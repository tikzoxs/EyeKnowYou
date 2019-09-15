import argparse
import os

import tensorflow as tf
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

# import videoto3d
from tqdm import tqdm

import train_generator as geny_tr
import validation_generator as geny_va
import test_generator as geny_te

LAYER_ACTIVATION = 'relu'

batch_size = 64


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

def main():
    global batch_size
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
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

    model.add(Conv3D(128, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid', name='new_dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', name='new_output'))

    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    # plot_model(model, show_shapes=True,
    #            to_file=os.path.join(args.output, 'model.png'))

    # checkpointer = ModelCheckpoint(filepath="/result_dir/weights.hdf5", verbose=1, save_best_only=True)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint("./weights.{epoch:02d}.hdf5",
    #                                       save_weights_only=True,
    #                                       verbose=1)

    result_direct = "/people/tkal976/aeye/result_dir/"
    # result_direct = "/home/tkal976/Desktop/Black/Codes/git/3DCNN/result_dir/"
    # model.load_weights(result_direct + "initial.hd5", by_name=True)
    

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=result_direct + "eye_know_you_weights_no_transfer.hd5",
                                          save_weights_only=True,
                                          verbose=1)    
    batch_size = args.batch
    train_gen = geny_tr.train_generator(batch_size)
    val_gen = geny_va.validation_generator(batch_size)
    test_gen = geny_te.test_generator(batch_size)

    '''np.floor(5275/batch_size)'''
    '''np.floor(339/batch_size)'''
    '''np.floor(544/batch_size)'''
    history = model.fit_generator(train_gen(), steps_per_epoch=np.floor(7450), epochs=args.epoch, callbacks=[cp_callback], validation_data=val_gen(),
        validation_steps=np.floor(600), class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=1)

    model.evaluate_generator(test_gen(), steps=np.floor(575), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'aeye_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'aeye_3dcnnmodel.hd5'))

    loss, acc = model.evaluate_generator(test_gen(), steps=8, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)


if __name__ == '__main__':
    main()