#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:16:23 2020

@author: Arnaud Gardille
"""

from __future__ import print_function
import numpy as np

import pickle
import gzip
import matplotlib.pyplot as plt
from time import time
from numpy.random import normal
from scipy.stats import norm

def sigmoid(x):
    return 1/(1+np.exp(-x))

class RBM:
  
    def __init__(self, num_visible, num_hidden, lr = 0.01):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True
        self.k = 10 # number of Gibbs sameling steps
        
        self.hidden_bias = np.zeros(num_hidden)
        self.visible_bias = np.zeros(num_visible)
        
        self.learningRate = lr
    
        self.L_errors = []
        self.L_b = []
        self.L_c = []
        
        self.cote = int(self.num_visible ** (1/2))
        
        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
        # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
        # standard deviation by multiplying the interval with appropriate value.
        # Here we initialize the weights with mean 0 and standard deviation 0.1. 
        
        
        # self.weights.shape == (num_hidden, num_visible)
        
    def probaHiddens(self, X):
        return 
    
    def SampleHiddens(self, X):
        p_h = self.probaHiddens(X)
        h = (np.random.random(self.num_hidden) < p_h)*1
        return h
    
    def probaVisible(self, H):        
        return 
    
    def SampleVisibles(self, H):
        p_v = self.probaVisible(H)
        v = (np.random.random(self.num_visible) < p_v)*1
        return v

    def gibbsSampling(self, X, k=1000):
        v = X
        for _ in range(self.k):
            h = self.SampleHiddens(v)
            v = self.SampleVisibles(h)
        return v
    
    def updateParameters(self, example, x_tilde):
        return
    
    def computeError(self, data):
        sum = 0
        for example in data:
            sum += np.sum((example - self.probaVisible(self.SampleHiddens(example))) ** 2)
        return sum*100/data.size

    def train(self, data, max_epochs = 1000):
        t0 = time()
        x_tilde = np.random.random(self.num_visible)
        
        for epoch in range(max_epochs):      
            for example in data :
                x_tilde = self.gibbsSampling(x_tilde, self.k)
                
                self.updateParameters(example, x_tilde)
                
            if self.debug_print:
                error = self.computeError(data)
                print("Epoch %s: error is %s" % (epoch, error))
                self.L_errors.append(error)
                self.param_save()
        
        print("durée d'entraiement : %s" % (time() - t0))
                
    def param_save(self):
        return
    
    def printError(self):
        X = np.arange(len(self.L_errors))
        plt.plot(X, self.L_errors)
        plt.title("Evolution de l'erreur")
        plt.show()
        
    def genImMoy(self, nb=100):
        ims = []
        for i in range(nb):
            v = np.random.random(self.num_visible)
            v = self.gibbsSampling(v, 1000)
            ims.append(v.reshape(self.cote, self.cote))
            
        im = np.sum(np.array(ims), axis=0)/100
        plt.title("moyenne de 100 images")
        plt.imshow(im)
        plt.show()
        
    def genPleinIm(self,  k=1000):
        
        f,ax = plt.subplots(10,10,figsize=(20,20))
        for i1 in range(10):
            print(i1)
            for i2 in range(10):
                v = np.random.random(self.num_visible)
                v = self.gibbsSampling(v, k)
                #print(v.shape)
                h = self.SampleHiddens(v)
                im = self.probaVisible(h)
                ax[i1,i2].imshow(im.reshape(self.cote, self.cote))
        plt.show()
        
    def affParamm(self):
        return
    
    def genIm(self, k=1000):
        v0 = np.random.random(self.num_visible)
        v = self.gibbsSampling(v0)
        
        h = self.SampleHiddens(v)
        im = self.probaVisible(h)
        
        plt.title("proba")
        plt.imshow(im.reshape(self.cote, self.cote))
        plt.show()
        
        plt.title("sample")
        plt.imshow(v.reshape(self.cote, self.cote))
        plt.show()
        
    def gen1Pixel(self, k=1000):
        v0 = np.random.random(self.num_visible)
        v = self.gibbsSampling(v0)
        
        h = self.SampleHiddens(v)
        im = self.probaVisible(h).reshape(self.cote, self.cote)
        
        plt.title("higher pixel")
        plt.imshow(im == np.max(im))
        plt.show()
        
    def probaGivNull(self):
        plt.title("Visible probability given a null hidden")
        plt.imshow(self.probaVisible(np.zeros(10)).reshape(self.cote, self.cote))
        plt.show()
        
    def Moyenne_EqType(self):
        X = np.zeros((10000, self.num_visible))
        for k in range(10000):
            v = np.random.random(self.num_visible)
            v = self.gibbsSampling(v, k)
            #h = self.SampleHiddens(v)
            #v = self.probaVisible(h)
            X[k] = self.gibbsSampling(v, k)

        plt.title("moyennes")
        M = np.sum(X, axis=0)
        print(M.shape)
        plt.imshow(M.reshape(self.cote, self.cote))
        plt.savefig("moy")
        plt.show()
        
        
        
        data = X.reshape(10000, self.cote, self.cote)
        plt.show()
        plt.title("equart type")
        plt.imshow(np.var(data, axis=0))
        plt.savefig("eq")
        plt.show()
        
class GBRBM_adapte(RBM):
    def __init__(self, num_visible, num_hidden, lr, useZ = True):
        super().__init__(num_visible, num_hidden, lr)
        self.printEvolParam = True
    
        np_rng = np.random.RandomState(1234)
    
        #self.visible_bias = np.zeros(num_visible) # TEST
        
        self.weights1 = np.asarray(np_rng.uniform(
    			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	size=(num_hidden, num_visible)))
        
        self.weights2 = np.asarray(np_rng.uniform(
    			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	size=(num_hidden, num_visible)))
        """
        self.weights1 = np.zeros((num_hidden, num_visible))* 0.001
        self.weights2 = np.zeros((num_hidden, num_visible))* 0.001
        """
        # self.weights.shape == (num_hidden, num_visible)
        
        #self.sigma = np.ones(num_visible)
        #self.L_sigma = []
        self.useZ = useZ
        
        if self.useZ:
            self.z = - np.ones(num_visible)
            self.L_z = []
        else:
            self.sigma = np.ones(num_visible) * 0.1
            self.L_sigma = []
            
        self.L_W1 = []
        self.L_W2 = []
        
        
    def probaHiddens(self, X):
        """
        p_h = np.zeros(self.num_hidden)
        for j in range(self.num_hidden):
            s1 = np.sum(self.weights1[j]*X*np.exp(-self.z))
            s2 = np.sum(self.weights2[j]*(X**2))
            p_h[j] = sigmoid( s1 + self.hidden_bias[j] + s2)
        return p_h
        """
        #return sigmoid( np.dot(self.weights1, X*self.sigma**(-2)) + self.hidden_bias + np.dot(self.weights2, X**2))
        if self.useZ:
            return sigmoid( np.dot(self.weights1, X*np.exp(-self.z)) + self.hidden_bias + np.dot(self.weights2, X**2))
        else:
            return sigmoid( np.dot(self.weights1, X*self.sigma**(-2)) + self.hidden_bias + np.dot(self.weights2, X**2))
    
    def probaVisible(self, H):
        """
        p_v = np.zeros(self.num_visible)
        for k in range(self.num_visible):
            D = self.calcDenom(H, k)
            moy = self.calcNumMoy(H, k) / D
            var = np.exp(self.z[k]) / D
            #print(norm.pdf(H[k], moy, var))
            p_v[k] = norm.rvs(moy, var)
        return p_v"""
        if self.useZ:
            D = 1 - 2*np.exp(self.z) *np.dot(H, self.weights2) 
        else:
            D = 1 - 2*(self.sigma**2)*np.dot(H, self.weights2) 
        moy = (np.dot(H, self.weights1) + self.visible_bias) / D
        return moy
    
    def SampleVisibles(self, H):
        if self.useZ:
            D = 1 - 2*np.exp(self.z) *np.dot(H, self.weights2) 
            moy = (np.dot(H, self.weights1) + self.visible_bias) / D
            var = np.exp(self.z / 2) / np.sqrt(D)
            return normal(moy, var)
        else:
            D = 1 - 2*(self.sigma**2) *np.dot(H, self.weights2) 
            moy = (np.dot(H, self.weights1) + self.visible_bias) / D
            var = self.sigma / np.sqrt(D)
            return normal(moy, var)
    def updateParameters(self, example, x_tilde):
        h_example = self.probaHiddens(example)
        h_tilde = self.probaHiddens(x_tilde)
        c = np.copy(self.visible_bias)
        #invVar = (self.sigma**(-2))
        
        if self.useZ:
            invVar = np.exp(-self.z)
        else:
            invVar = self.sigma**(-2)
            
        if np.any(invVar != invVar):
            print(invVar)
            raise
            
        self.hidden_bias += self.learningRate * (h_example - h_tilde)
        self.visible_bias += self.learningRate * invVar * (example - x_tilde)
        
        s1 = (example - c)**2 - (x_tilde - c)**2
        s2 = example * np.dot(self.weights1.T, h_example) - x_tilde * np.dot(self.weights1.T, h_tilde)
        if self.useZ:
            self.z += self.learningRate *  0.01 * invVar * (0.5*s1 - s2)
            
            self.z = np.maximum(self.z, -100)
            self.z = np.minimum(self.z, 1)
        else:
            self.sigma += self.learningRate * 0.01 * (self.sigma**(-3)) * (s1 - 2*s2)
            
            self.sigma = np.maximum(self.sigma, 0)
            self.sigma = np.minimum(self.sigma, 3)

        self.weights1 += self.learningRate * (np.outer(h_example, invVar * example) - np.outer(h_tilde, invVar * x_tilde))
        self.weights2 += self.learningRate * (np.outer(h_example, (example**2)) - np.outer(h_tilde, (x_tilde**2)))
    
    def param_save(self):
        self.L_W1.append(np.copy(self.weights1))
        self.L_W2.append(np.copy(self.weights2))
        if self.useZ:
            self.L_z.append(np.copy(self.z))
        else:
            self.L_sigma.append(np.copy(self.sigma))
        self.L_b.append(np.copy(self.hidden_bias))
        self.L_c.append(np.copy(self.visible_bias))
        
    def affParamm(self):
        print("c : max %s, min %s" % (max(self.visible_bias), min(self.visible_bias)))
        print("b : max %s, min %s" % (max(self.hidden_bias), min(self.hidden_bias)))
        if self.useZ:
            print("z : max %s, min %s" % (max(self.z), min(self.z)))
        else:
            print("sigma : max %s, min %s" % (max(self.sigma), min(self.sigma)))

        plt.title("W_1")
        plt.ylabel("hidden neurons")
        plt.xlabel("visible neurons")
        plt.imshow(self.weights1)
        plt.savefig("W_1")
        plt.show()
        
        plt.title("W_2")
        plt.ylabel("hidden neurons")
        plt.xlabel("visible neurons")
        plt.imshow(self.weights2)
        plt.savefig("W_2")
        plt.show()
 
        if self.useZ:
            plt.title("Z")
            plt.ylabel("visible neurons")
            plt.xlabel("Epoch of training")
            plt.imshow(np.array(self.L_z).T)
            plt.savefig("Z")
            plt.show()
        else:
            plt.title("sigma")
            plt.ylabel("visible neurons")
            plt.xlabel("Epoch of training")
            plt.imshow(np.array(self.L_sigma).T)
            plt.savefig("sigma")
            plt.show()
            
        plt.title("hidden bias")
        plt.ylabel("hidden neurons")
        plt.xlabel("Epoch of training")
        plt.imshow(np.array(self.L_b).T)
        plt.savefig("HB")
        plt.show()
        
        plt.title("visible bias")
        plt.ylabel("visible neurons")
        plt.xlabel("Epoch of training")
        plt.imshow(np.array(self.L_c).T)
        plt.savefig("VB")
        plt.show()
      
class BRBM(RBM):
    def __init__(self, num_visible, num_hidden, lr):
        super().__init__( num_visible, num_hidden, lr)
        
        np_rng = np.random.RandomState(1234)
    
    
        self.weights = np.asarray(np_rng.uniform(
    			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	size=(num_hidden, num_visible)))
        
        
    def probaHiddens(self, X):
        """
        p_h = np.zeros(self.num_hidden)
        for j in range(self.num_hidden):
            p_h[j] = sigmoid(self.hidden_bias[j] + np.dot(self.weights[j], X))
        return p_h
        """
        return sigmoid(self.hidden_bias + np.dot(self.weights, X))
    
    def SampleHiddens(self, X):
        p_h = self.probaHiddens(X)
        h = (np.random.random(self.num_hidden) < p_h)*1
        return h
    
    def probaVisible(self, H):
        """
        p_v = np.zeros(self.num_visible)
        for k in range(self.num_visible):
            p_v[k] = sigmoid(self.visible_bias[k] + np.dot(self.weights[:,k], H))
        return p_v
        """
        
        return sigmoid(self.visible_bias + np.dot(self.weights.T, H))
    
    def SampleVisibles(self, H):
        p_v = self.probaVisible(H)
        v = (np.random.random(self.num_visible) < p_v)*1
        return v
    
    def updateParameters(self, example, x_tilde):
        h_example = self.probaHiddens(example)
        h_tilde = self.probaHiddens(x_tilde)
        
        """
        for j in range(self.num_hidden):
            for k in range(self.num_visible):
                self.weights[j][k] += self.learningRate * (h_example[j]*example[k] - h_tilde[j]*x_tilde[k])
                
        for j in range(self.num_hidden):
            self.hidden_bias[j] += self.learningRate * (h_example[j] - h_tilde[j])
            
        for k in range(self.num_visible):
            self.visible_bias[k] += self.learningRate * (example[k] - x_tilde[k])
        """
        
        self.weights += self.learningRate * (np.outer(h_example, example) - np.outer(h_tilde, x_tilde))
        self.hidden_bias += self.learningRate * (h_example - h_tilde)
        self.visible_bias += self.learningRate * (example - x_tilde)
        
    def affParamm(self):
        print("c : max %s, min %s" % (max(self.visible_bias), min(self.visible_bias)))
        print("b : max %s, min %s" % (max(self.hidden_bias), min(self.hidden_bias)))
        
        plt.title("W")
        plt.ylabel("hidden neurons")
        plt.xlabel("visible neurons")
        plt.imshow(self.weights)
        plt.savefig("W")
        plt.show()
        
class GBRBM_Z(RBM):
  
    def __init__(self, num_visible, num_hidden, lr):
        super().__init__(num_visible, num_hidden, lr)
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
    			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	size=(num_hidden, num_visible)))
        'self.sigma = np.ones(num_visible) * 0.001'
        
        #self.visible_bias = np.zeros(num_visible) # TEST
        
        self.z = np.zeros(num_visible)
        
        self.L_z = []

    
    def probaHiddens(self, X):
        res = sigmoid(self.hidden_bias + np.dot(self.weights ,X*np.exp(- self.z)))
        if np.any(res != res):
            print(self.z)
            print(res)
            raise
        return res
    
    def probaVisible(self, H):
        moy = np.dot(H ,self.weights) + self.visible_bias  
        return moy
    
    def SampleVisibles(self, H):
        moy = np.dot(H ,self.weights) + self.visible_bias  
        return normal(moy, np.exp(self.z / 2))
    
    def updateParameters(self, example, x_tilde):
        h_example = self.probaHiddens(example)
        h_tilde = self.probaHiddens(x_tilde)
        invVar = np.exp(- self.z)
        if np.any(invVar != invVar):
            raise 
    
        self.hidden_bias += self.learningRate * (h_example - h_tilde)
        
        
        
        s1 = (example - self.visible_bias)**2 - (x_tilde - self.visible_bias)**2
        s2 = example * np.dot(h_example, self.weights) - x_tilde * np.dot(h_tilde, self.weights)
        self.z += self.learningRate * invVar * (s1 - 2*s2) # OU O.5
        
        self.z = np.maximum(self.z, -100)
        self.z = np.minimum(self.z, 1)
        
        self.visible_bias += self.learningRate * invVar * (example - x_tilde)
        
        self.weights += self.learningRate * (np.outer(h_example, invVar * example) - np.outer(h_tilde, invVar * x_tilde))
       
    def param_save(self):
        self.L_z.append(np.copy(self.z))
        self.L_b.append(np.copy(self.hidden_bias))
        self.L_c.append(np.copy(self.visible_bias))
         
    def affParamm(self):
        print("c : max %s, min %s" % (max(self.visible_bias), min(self.visible_bias)))
        print("b : max %s, min %s" % (max(self.hidden_bias), min(self.hidden_bias)))
        print("z : max %s, min %s" % (max(self.z), min(self.z)))
        
        plt.title("W")
        plt.xlabel("visible neurons")
        plt.ylabel("hidden neurons")
        plt.imshow(self.weights)
        plt.savefig("W")
        plt.show()
        
        plt.title("Visible bias")
        plt.xlabel("epoch of training")
        plt.ylabel("visible neurons")
        plt.imshow(np.array(self.L_c).T)
        plt.savefig("VB")
        plt.show()
        
        plt.title("Hidden bias")
        plt.xlabel("epoch of training")
        plt.ylabel("Hidden neurons")
        plt.imshow(np.array(self.L_c).T)
        plt.savefig("HB")
        plt.show()
        
        plt.title("z : log of sigma^2")
        plt.xlabel("Epoch of training")
        plt.ylabel("visible neurons")
        plt.imshow(np.array(self.L_z).T)
        plt.savefig("Z")
        plt.show()

    
if __name__ == '__main__':
    
    """
    r = RBM(num_visible = 6, num_hidden = 2)
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    
    print(r.SampleHiddens(training_data[0]))
    print(r.probaHiddens(training_data[0]))
    print()
    print(training_data[0])
    print(r.SampleVisibles(r.SampleHiddens(training_data[0])))
    
    r.train(training_data)
    
    print()
    print(training_data[0])
    print(r.SampleVisibles(r.SampleHiddens(training_data[0])))
    
    ---
    r.train(training_data, max_epochs = 5000)
    print(r.weights)
    user = np.array([[0,0,0,1,1,0]])
    print(r.run_visible(user))
    
    user = np.array([[0,0,0,1,1,1]])
    print(r.run_visible(user))
    
    user = np.array([[1,1,0,0,1,0]])
    print(r.run_visible(user))
    
    user = np.array([[1,1,0,0,1,0]])
    print(r.run_visible(user))
    """
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    train_set, _, _ = p
    
    X = train_set[0][:100,:].T
     
    r = RBM(num_visible = 784, num_hidden = 500)
    training_data = X.T
    #r.train(training_data, max_epochs = 10)
    r.trainWithBatch(training_data, max_epochs = 100, bs = 5)
    r.printError()

    im = X[:,0]
    plt.title("Image d'origine")
    plt.imshow(im.reshape(28,28))
    plt.show()
    
    r.genIm()
    r.affParamm()
    """
    t0 = time()
    
    X = np.load('data.npy')
    (nb, cote, _) = X.shape
    size = cote**2
    training_data = X.reshape(nb, size)
    
    nbHidden = 1
    lr = 0.0001
    nbEpoch = 100
    """
   
    print("BRBM")
    r1 = BRBM(size, nbHidden, lr)
    r1.train(training_data, max_epochs = nbEpoch)
    r1.printError()
    #r1.genImMoy()
    r1.Moyenne_EqType()
    r1.affParamm()


    #r1.genIm()  
    #r1.gen1Pixel()
    #r1.genPleinIm()
    


    print("GBRBM_Z")
    r2 = GBRBM_Z(size, nbHidden, lr)
    r2.train(training_data, max_epochs = nbEpoch)
    r2.printError()
    #r2.genImMoy()
    r2.Moyenne_EqType()
    r2.affParamm()
    

    #r2.genIm()  
    #r2.gen1Pixel()
    #r2.genPleinIm()

    
    print("GBRBM_adapte_sigma")
    r4 = GBRBM_adapte(size, nbHidden, lr, False)
    r4.train(training_data, max_epochs = nbEpoch)
    r4.printError()
    #r3.genImMoy()
    r4.Moyenne_EqType()
    
    r4.affParamm()

    
    
    """
    print("GBRBM_adapte_Z")
    r3 = GBRBM_adapte(size, nbHidden, lr, True)
    r3.train(training_data, max_epochs = nbEpoch)
    r3.printError()
    #r3.genImMoy()
    r3.Moyenne_EqType()
    
    r3.affParamm()
    

    #r3.genIm()  
    #r3.gen1Pixel()
    #r3.genPleinIm()

    
    
    
    
    