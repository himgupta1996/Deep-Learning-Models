# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:44:39 2019

@author: Himanshu
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import optimizers

def unpickle(file):
    import pickle
    print("teh file is %s" % ((file)))
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

X_train = []
Y_train = []
for i in range(1,6):
    batch = unpickle("D:\Study\Data Science\Deep Learning Course - Iisc CCE\Assignment-2\cifar-10-batches-py\data_batch_"+str(i))
    X_train_batch = batch[b'data']
    Y_train_batch = batch[b'labels']
    for x in X_train_batch:
        X_train.append(x)
    for y in Y_train_batch:
        Y_train.append(y)
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)

##Ask Doubt
test_batch = unpickle("D:\Study\Data Science\Deep Learning Course - Iisc CCE\Assignment-2\cifar-10-batches-py\\test_batch")
X_test = test_batch[b'data']
Y_test = test_batch[b'labels']

X_val = X_train[42000:]
Y_val = Y_train[42000:]
X_train = X_train[0:42000]
Y_train = Y_train[0:42000]

num_classes = 10

X_train = X_train.reshape([42000,3,32,32])
X_val = X_val.reshape([8000,3,32,32])
X_test = X_test.reshape([10000,3,32,32])

X_train = (X_train.astype('float32'))/255.0
X_val = (X_val.astype('float32'))/255.0
X_test = (X_test.astype('float32'))/255.0

Y_train = to_categorical(Y_train, num_classes)
Y_val = to_categorical(Y_val, num_classes)
Y_test = to_categorical(Y_test, num_classes)

epoch = 50
learning_rates = [0.05]
batch_sizes = [128,256]
fig, axes = plt.subplots(nrows=len(batch_sizes), ncols=2, figsize = (25,5*(len(batch_sizes))))
plot_index = 0
for batch_size in batch_sizes:
    for lr in learning_rates:
        ##Create and compile the model`
        cnn = Sequential()
        cnn.add(Conv2D(32, kernel_size=(3,3), input_shape=(3,32,32), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        cnn.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        cnn.add(Flatten())
        cnn.add(Dense(128,activation='relu'))
        cnn.add(Dense(10,activation='softmax'))
        
        sgd = optimizers.SGD(lr=lr)
        cnn.compile(loss='mean_squared_error', optimizer = sgd, metrics=['mse', 'accuracy']) 
        
        cc = cnn.fit(X_train, Y_train, epochs=epoch, batch_size = batch_size, verbose=1, validation_data=(X_val, Y_val))
        
        Accuracy_validation = cc.history['val_acc']
        Accuracy_training = cc.history['acc']
        test_accuracy = cnn.evaluate(X_test, Y_test)
        print("The testing accuracy metric is %s" % (test_accuracy))
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