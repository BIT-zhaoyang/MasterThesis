# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:54:02 2016

@author: yangzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from scipy.stats import multivariate_normal as multiGaussian
from numpy import newaxis

def generate_data():
    numPoint = 1000
    numComponent = 3
    numDim = 2
    
    # transition matrix and emission parameters are given by predifined
    A = np.array([[0.8, 0.1, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.1, 0.1, 0.8]])
    mean = np.array([ [-2, 0],
                      [3, 0],
                      [7, 3] ])
    cov = np.array([ [[1, 1],
                      [1, 3]],
                     [[1, -1],
                      [-1, 3]],
                     [[1, 1],
                      [1, 3]] ])
    
    # generate data
    # use the uniform distribution for the choosing the first component
    Z = np.zeros((numPoint), dtype = np.int8)
    X = np.zeros((numPoint, numDim))
    
    Z[0] = np.random.uniform(low = 0, high = numComponent)
    X[0] = np.random.multivariate_normal(mean[Z[0]], cov[Z[0]])
    
    for i in range(1, numPoint):
        last = Z[i-1]
        curr = np.random.choice(numComponent, 1, p = A[last])
        Z[i] = curr
        X[i] = np.random.multivariate_normal(mean[Z[i]], cov[Z[i]])
    
    return X, Z

class KMeans():
    def __init__(self, X, Z):
        """ import data and set parameters"""
        self.X = X
        self.nPoint, self.nDim = X.shape
        self.nComp = np.unique(Z).size
        """ initialize \mu_k """
        randIdx = np.random.random_integers(low=0, high=self.nPoint, size=self.nComp)
        self.mean = X[randIdx, :]
        self.Y = np.zeros((self.nPoint, self.nComp))
    
    def EStep(self):
        for i in range(self.nPoint):
            ele = self.X[i]
            dist = ele[newaxis, :] - self.mean
            dist = np.sqrt((dist * dist).sum(axis=1))
            self.Y[i] = np.zeros(self.nComp)
            self.Y[i][np.argmin(dist)] = 1
    
    def MStep(self):
        total = self.Y.T.dot(self.X)
        respons = self.Y.sum(axis=0)
        self.mean = total / respons[:, newaxis]
        
    def getAssign(self):
        return self.Y
    
    def getMean(self):
        return self.mean
        
    def fullEM(self):
        self.EStep()
        self.MStep()
        return self.getAssign(), self.getMean()
        
if __name__ == "__main__":
    X, Z = generate_data()
    plt.figure(figsize = (12, 18))
    # plot original data    
    plt.subplot(321)
    plt.scatter(X[:, 0], X[:, 1], c = Z)
    plt.title("(a) Original data and corresponding classes")
    
    kmeans = KMeans(X, Z)

    # E step in first iteration
    kmeans.EStep()
    Y = kmeans.getAssign()
    mean = kmeans.getMean()
    plt.subplot(322)
    plt.scatter(X[:, 0], X[:, 1], c = Y)
    plt.scatter(mean[:, 0], mean[:, 1], s = 50, color = 'y', marker = 'D')
    plt.title("(b) E Step in the first iteration")
    
    # M step in first iteration
    kmeans.MStep()
    Y = kmeans.getAssign()
    mean = kmeans.getMean()
    plt.subplot(323)
    plt.scatter(X[:, 0], X[:, 1], c = Y)
    plt.scatter(mean[:, 0], mean[:, 1], s = 50, color = 'y', marker = 'D')
    plt.title("(c) M Step in the first iteration")
    
    # full EM 
    Y, mean = kmeans.fullEM()
    plt.subplot(324)
    plt.scatter(X[:, 0], X[:, 1], c = Y)
    plt.scatter(mean[:, 0], mean[:, 1], s = 50, color = 'y', marker = 'D')
    plt.title("(d) Results after two full EM iterations")
    
    # 5 iterations
    for i in range(3):
        Y, mean = kmeans.fullEM()
    plt.subplot(325)
    plt.scatter(X[:, 0], X[:, 1], c = Y)
    plt.scatter(mean[:, 0], mean[:, 1], s = 50, color = 'y', marker = 'D')
    plt.title("(e) Results after five full EM iterations")
    
    # converged
    for i in range(15):
        Y, mean = kmeans.fullEM()
    plt.subplot(326)
    plt.scatter(X[:, 0], X[:, 1], c = Y)
    plt.scatter(mean[:, 0], mean[:, 1], s = 50, color = 'y', marker = 'D')
    plt.title("(f) Results after convergence")