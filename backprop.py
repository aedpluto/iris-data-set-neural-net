# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:57:05 2017

"""

import numpy as np
from sklearn import datasets

class NeuralNetwork(object):
    def __init__(self):
        self.input_layer_size = 4
        self.output_layer_size = 3
        self.W = np.random.randn(self.input_layer_size, self.output_layer_size)
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def forward(self, X):
        z = np.dot(X, self.W)
        a = self.sigmoid(z)
        return z, a
    
    def backward(self, X, Y, iterations=2000, learning_rate=0.01):
        for i in range(iterations):
            weighted_sum_of_inputs, actual_output = self.forward(X)
            delta = np.multiply((actual_output - Y),
                                self.sigmoid_prime(weighted_sum_of_inputs))
            delta_weighted = np.dot(X.T, delta)
            self.W -= learning_rate * delta_weighted
            
    def predict(self, X):
        return self.forward(X)[1]
    

def example():
    # preprocessing
    iris = datasets.load_iris()
    X = np.concatenate((iris.data[:48], iris.data[50:98], iris.data[100:148]), axis=0)
    test = np.concatenate((iris.data[48:50], iris.data[98:100], iris.data[148:150]), axis=0)
    
    Y = []
    options = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
    for c in range(3):
        for i in range(int(len(X)/3)):
            Y.append(options[c])
    Y = np.array(Y)
    
    # initialize
    NN = NeuralNetwork()
    
    # training
    NN.backward(X, Y, iterations=5000)
    
    # testing
    hypothesis = NN.predict(test)
    for line in hypothesis:
        l = [line[0], line[1], line[2]]
        i = l.index(max(l))
        if i == 0:
            print('Setosa')
        if i == 1:
            print('Versicolor')
        if i == 2:
            print('Virginica')
                
example()
