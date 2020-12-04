# -*- coding: utf-8 -*-
"""

Created on Mon Feb 18 19:17:19 2019

@author: Himanshu Gupta

VARIABLES:
    D => No. of Features in input matrix
    X => Input matrix(N x D+1)
    Y => Label vector(N x 1)
    T => Taget Matrix(N x K)
    W => Weight Matrix(D+1, K)
    softmax_a = (Y_predicted) => Prediction Matrix(N x K)
    batch_count => No. of batches in one epoch
    batch_size => No. of taining examples in one mini batch of one epoch.
INSTRUCTION:
    1. Enter the value of epoch.
    2. Enter the values of learning rates in the array "learning_rates".
    3. Enter the values of batch counts int the array "batch_counts"
OUTPUT:
    1. Test Accuracy after last epoch
    2. Graph of Validation Accuracy vs Epoch.
    3. Graph of Training Accuracy vs Epoch.
    4. Graph of Validation Likelihood vs Epoch.
    5. Graph of Training Likelihood vs Epoch.
"""
import sys
import time
from math import log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
mnist_data = pd.read_csv("train.csv")
X = mnist_data.drop(["label"], axis=1)
Y = mnist_data["label"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)

X_train = X_train.values
#X_val = (np.c_[np.ones(len(X_val)), X_val.values])/255
##normalizing the X_val matrix and adding bias
X_val = (X_val.values)/255
X_val = np.c_[np.ones(len(X_val)), X_val]
X_test = (X_test.values)/255
X_test = np.c_[np.ones(len(X_test)), X_test]
Y_train = Y_train.values
Y_val = Y_val.values
Y_test = Y_test.values

mnist_train_data = np.c_[Y_train, X_train]

##getting no of training example (N) and coulmns (D)
N = len(X_train)
D = len(X_train[0,:])

#no of classes
K = 10

epoch = 100
batch_counts = [N]
learning_rates = [ 0.1]

def epochProgressBar(value, endvalue, bar_length=epoch, title="EPOCH"):
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}%".format(title, arrow + spaces, int(round(percent * 100))))
    if value == endvalue:
        sys.stdout.write("\n")
    else:
        sys.stdout.flush()

def multiply_matrix(A, B):
    ##let suppose the dimension of A is mxn
    ##let suppose the dimension of B is nxo
    ##Let the new matrix be C
    (m,n) = A.shape
    (n,o) = B.shape
    C = np.zeros((m,o))
    for i in range(m):
        for j in range(o):
            sum = 0
            for k in range(n):
                sum+=A[i][k]*B[k][j]
            C[i][j] = sum
    return C

def softmax(np_object, matrix = False, axis = 0):
    
    if not matrix:
        np_object = np.exp(np_object)
        sum_np_object = sum(np_object)
        np_object = np_object/sum_np_object
    else:
        if axis == 0:
            np_object = np.exp(np_object)
            for i in range(np_object.shape[0]):
                np_object[i] = np_object[i]/sum(np_object[i])
        elif axis == 1:
            np_object = np.exp(np_object)
            for j in range(np_object.shape[1]):
                np_object[:,j] = np_object[:,j]/sum(np_object[:,j])
    return np_object

def calculate_accuracy(X, Y, W):
    TP_count=0
    a = np.dot(X, W)
    softmax_a = softmax(a, matrix = True, axis = 0)
    for i in range(len(X)):
        label = np.argmax(softmax_a[i])
        if label == Y[i]:
            TP_count+=1
    acc = (TP_count/ len(X))*100
    return acc

def calculate_loglikelihood(X, T, W):
    a = np.dot(X, W)
    Y = softmax_a = softmax(a, matrix = True, axis = 0)
    log_likelihood = 0
    for i in range(len(T)):
        for j in range(len(T[i])):
            log_likelihood += (T[i][j] * log(Y[i][j]))
    return log_likelihood
    

Errors = []
Error_val = []

##doing stochastic gradient descent for multiple epochs
def perform_sgd():
#    global W
    fig, axes = plt.subplots(nrows=2*len(batch_counts), ncols=2, figsize = (25,45))
    plot_index = 0
    
    for batch_count in batch_counts:
        batch_size = int(N/batch_count)
        #plt.subplot(len(batch_counts),1, plot_no)
        for learning_rate in learning_rates:
            ##initializing W matrix with random numbers between -1 and 1
            print("----------For Batch Size = %s, Learning Rate = %s----------" % (batch_count, learning_rate))
            
            ##Initializing weight matrix
            W = np.random.rand(D+1,K)
            
            ##Initializing Validation accuracy array for a particular learning rate on profress of epoch
            Accuracy_validation = []
            Accuracy_training = []
            Log_likelihood_validation = []
            Log_likelihood_training = []
            
            start_time = time.time()
            
            ##For tracking the completed epochs
            epoch_count = 0
            for i in range(epoch):
                ##Shuffling the data set
                np.random.shuffle(mnist_train_data)
                
                ##Working on gradient descent on mini batches of the whole data(one epoch)
                
                X_train = (mnist_train_data[:,1:])/255 ##normalizing the X matrix
                Y_train = mnist_train_data[:,0]
                T_train = np.zeros((len(Y_train),K))
                for i in range(len(T_train)):
                    T_train[i][Y_train[i]] = 1
                ##Adding bias to X_train matrix
                X_train = np.c_[np.ones(N), X_train]
                
                ##for selecting next batch
                index = 0
                
                ##for tracking completed batch in one epoch
                count = 0
                
                for j in range(batch_count):
                    
                    X_batch = X_train[index:index+batch_size,:]
                    Y_batch = Y_train[index:index+batch_size]
                    
                    ##initializing T matrix
                    T_batch = np.zeros((batch_size,K))
                    for i in range(batch_size):
                        T_batch[i][Y_batch[i]] = 1
                    
                    ##Calulating the softmax
                    a = np.dot(X_batch, W)
                    softmax_a = softmax(a, matrix = True, axis = 0)                
                    
                    ##calculating the difference between predicted and target value of label
                    ##Hence calculating error
                    diff = T_batch - softmax_a
                    
                    ##mutiplying it with X matrix transpose
                    gradient = (1/batch_size) * (np.dot(X_batch.transpose(), diff))                   
                    
                    ##updating the W matrix
                    W += learning_rate * gradient
                    
                    index = index + batch_size
                    count +=1
                
                
                ##calculating Training Accuracy
                train_acc = calculate_accuracy(X_train, Y_train, W)
                Accuracy_training.append(train_acc)
                
                ##calculating log likelihood of training set
                log_likelihood = calculate_loglikelihood(X_train, T_train, W)
                Log_likelihood_training.append(log_likelihood)
                
                T_val = np.zeros((len(Y_val),K))
                for i in range(len(T_val)):
                    T_val[i][Y_val[i]] = 1
                    
                ##calculating log likelihood of validation set
                log_likelihood = calculate_loglikelihood(X_val, T_val, W)
                Log_likelihood_validation.append(log_likelihood)
                
                ##calculating Validation Accuracy
                val_acc = calculate_accuracy(X_val, Y_val, W)
                Accuracy_validation.append(val_acc)
                
                epochProgressBar(epoch_count+1, epoch)
                epoch_count+=1
        
            end_time = time.time()
            
            ##Total time taken for all epochs for one learning rate
            time_taken = (end_time - start_time)/60
            
            ##caculating Testing Accuracy
            test_acc = calculate_accuracy(X_test, Y_test, W)
            print("Testing Accuracy is : %s%%" % (test_acc))
            
            axes[plot_index][0].plot(np.arange(1,epoch+1),Accuracy_validation,label="n=%s"%(learning_rate))
            axes[plot_index][1].plot(np.arange(1,epoch+1),Accuracy_training,label="n=%s"%(learning_rate))
            axes[plot_index+len(batch_counts)][0].plot(np.arange(1,epoch+1),Log_likelihood_validation,label="n=%s"%(learning_rate))
            axes[plot_index+len(batch_counts)][1].plot(np.arange(1,epoch+1),Log_likelihood_training,label="n=%s"%(learning_rate))
            
        axes[plot_index][0].grid(True)
        axes[plot_index][0].set_xlabel("--- Epochs --->")
        axes[plot_index][0].set_ylabel("--- Validation Accuracy --->")
        axes[plot_index][0].set_title("Batch size = %s" % (batch_count))
        axes[plot_index][0].legend()
        
        axes[plot_index][1].grid(True)
        axes[plot_index][1].set_xlabel("--- Epochs --->")
        axes[plot_index][1].set_ylabel("--- Training Accuracy --->")
        axes[plot_index][1].set_title("Batch size = %s" % (batch_count))
        axes[plot_index][1].legend()
        
        axes[plot_index+len(batch_counts)][0].grid(True)
        axes[plot_index+len(batch_counts)][0].set_xlabel("--- Epochs --->")
        axes[plot_index+len(batch_counts)][0].set_ylabel("--- Validation Log Likelihood --->")
        axes[plot_index+len(batch_counts)][0].set_title("Batch size = %s" % (batch_count))
        axes[plot_index+len(batch_counts)][0].legend()
        
        axes[plot_index+len(batch_counts)][1].grid(True)
        axes[plot_index+len(batch_counts)][1].set_xlabel("--- Epochs --->")
        axes[plot_index+len(batch_counts)][1].set_ylabel("--- Training Log Likelihood --->")
        axes[plot_index+len(batch_counts)][1].set_title("Batch size = %s" % (batch_count))
        axes[plot_index+len(batch_counts)][1].legend()

        plot_index +=1
    
    #fig.savefig('Experiment2.pdf')
    fig.savefig('Experiment2.jpg')
    
if __name__=="__main__":
    perform_sgd()
    
    