# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:15:38 2019

@author: Himanshu
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

val_images = train_images[50000:]
val_labels = train_labels[50000:]
train_images = train_images[0:50000]
train_labels = train_labels[0:50000]

train_images = train_images / 255.0
test_images = test_images / 255.0

num_classes = 10

momentum = [0.9, 0.95, 0.99]
epoch = 15

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (25,5))
plot_index=0
for momt in momentum:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)])
    
    sgd = optimizers.SGD(momentum=momt)
    model.compile(loss='mean_squared_error', optimizer = sgd, metrics=['mse', 'accuracy'])    
    #model.compile(loss='mean_squared_error', optimizer='sgd')
    y_train = keras.utils.to_categorical(train_labels, num_classes)
    y_test = keras.utils.to_categorical(test_labels, num_classes)
    y_val = keras.utils.to_categorical(val_labels, num_classes)
    cc = model.fit(train_images, y_train, epochs=epoch,
              validation_data=(val_images, y_val))
    Accuracy_validation = cc.history['val_acc']
    Accuracy_training = cc.history['acc']
    test_accuracy = model.evaluate(test_images, y_test)
    print("The testing accuracy metric for momentum %s is : %s" % (momt, test_accuracy))
    print("---------------------------------------------------")
    axes[0].plot(np.arange(1,epoch+1),Accuracy_validation,label="momt.=%s"%(momt))
    axes[1].plot(np.arange(1,epoch+1),Accuracy_training,label="momt.=%s"%(momt))
axes[0].grid(True)
axes[0].set_xlabel("--- Epochs --->")
axes[0].set_ylabel("--- Validation Accuracy --->")
axes[0].legend()

axes[1].grid(True)
axes[1].set_xlabel("--- Epochs --->")
axes[1].set_ylabel("--- Training Accuracy --->")
axes[1].legend()
fig.savefig('DNN_momentum_variation_final.jpg')
