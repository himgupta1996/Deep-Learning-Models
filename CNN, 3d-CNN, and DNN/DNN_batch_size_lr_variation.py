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
val_images = val_images / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(train_labels, num_classes)
y_test = keras.utils.to_categorical(test_labels, num_classes)
y_val = keras.utils.to_categorical(val_labels, num_classes)

learning_rates = [0.001, 0.01, 0.05, 0.1]
batch_sizes = [1, 32, 128, 1024]
epoch = 15

fig, axes = plt.subplots(nrows=len(batch_sizes), ncols=2, figsize = (25,25))
plot_index=0
for batch_size in batch_sizes:
    for lr in learning_rates:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)])
        
        
        sgd = optimizers.SGD(lr=lr, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer = sgd, metrics=['mse', 'accuracy'])   
        #model.compile(loss=' mean_squared_error', optimizer='sgd')
        cc = model.fit(train_images, y_train, epochs=epoch, batch_size=batch_size,
                  validation_data=(val_images, y_val))
        Accuracy_validation = cc.history['val_acc']
        Accuracy_training = cc.history['acc']
        test_accuracy = model.evaluate(test_images, y_test)
        print("The testing accuracy metric for lr %s and batch_size %s is %s" % (lr, batch_size, test_accuracy))
        print("---------------------------------------------------")
        axes[plot_index][0].plot(np.arange(1,epoch+1),Accuracy_validation,label="lr=%s"%(lr))
        axes[plot_index][1].plot(np.arange(1,epoch+1),Accuracy_training,label="lr=%s"%(lr))

    axes[plot_index][0].grid(True)
    axes[plot_index][0].set_xlabel("--- Epochs --->")
    axes[plot_index][0].set_ylabel("--- Validation Accuracy --->")
    axes[plot_index][0].set_title("Batch size = %s" % (batch_size))
    axes[plot_index][0].legend()
    
    axes[plot_index][1].grid(True)
    axes[plot_index][1].set_xlabel("--- Epochs --->")
    axes[plot_index][1].set_ylabel("--- Training Accuracy --->")
    axes[plot_index][1].set_title("Batch size = %s" % (batch_size))
    axes[plot_index][1].legend()
    plot_index+=1
fig.savefig('DNN_batch_size_lr_variation_final.jpg')
