# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:07:26 2019

@author: Himanshu
"""

#import tensorflow as tf
#import keras
#from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

num_classes = 10

val_images = train_images[50000:]
val_labels = train_labels[50000:]
train_images = train_images[0:50000]
train_labels = train_labels[0:50000]

train_images = train_images.reshape([50000,28,28,1])
val_images = val_images.reshape([10000,28,28,1])
test_images = test_images.reshape([10000,28,28,1])

train_images = (train_images.astype('float32'))/255.0
val_images = (val_images.astype('float32'))/255.0
test_images = (test_images.astype('float32'))/255.0

train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

learning_rate = 0.01
batch_size = 128
epoch = 15
cnn_layers = [1,2]

def build_cnn(cnn):
    cnn.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    return cnn

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (25,5))
plot_index=0
for no_layers in cnn_layers:
    cnn = Sequential()
    cnn.add(Conv2D(128, kernel_size=(3,3), input_shape=(28,28,1), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    for i in range(no_layers-1):
        cnn = build_cnn(cnn)
    cnn.add(Flatten())
    cnn.add(Dense(512,activation='relu'))
    cnn.add(Dense(10,activation='softmax'))
    cnn.compile(loss='mean_squared_error', optimizer = 'sgd', metrics=['mse', 'accuracy'])
    #cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['mse', 'accuracy'])
    
    history_cnn = cnn.fit(train_images, train_labels, epochs=epoch, verbose=1, validation_data=(val_images, val_labels))
    
    Accuracy_validation = history_cnn.history['val_acc']
    Accuracy_training = history_cnn.history['acc']
    
    test_accuracy = cnn.evaluate(test_images, test_labels)
    print("The testing accuracy metric for %s no. of cnn layers is %s" % (no_layers, test_accuracy))
    print("---------------------------------------------------")
    
    axes[0].plot(np.arange(1,epoch+1),Accuracy_validation,label="cnn_layers=%s"%(no_layers))
    axes[1].plot(np.arange(1,epoch+1),Accuracy_training,label="cnn_layers=%s"%(no_layers))

axes[0].grid(True)
axes[0].set_xlabel("--- Epochs --->")
axes[0].set_ylabel("--- Validation Accuracy --->")
axes[0].legend()

axes[1].grid(True)
axes[1].set_xlabel("--- Epochs --->")
axes[1].set_ylabel("--- Training Accuracy --->")
axes[1].legend()

fig.savefig('Experiment_cnn_layers_final.jpg')
