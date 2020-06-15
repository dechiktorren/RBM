#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:05:55 2020

@author: macbookair
"""
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

"""
Création de données, cad de 5000 images de 5*5.
la moyenne augment de droite a gauche
l'équart type augmente de haut en bas 
"""

# Generation de données
nb = 5000
size = 5

data = np.zeros((nb, size, size))

for i in range(size):
    for j in range(size):
        data[:,i, j] = normal(i/size, 0.3*j/size, nb)


plt.title("moyennes")
M = np.sum(data, axis=0)
plt.imshow(M)
plt.show()

plt.title("equart type")
plt.imshow(np.var(data, axis=0))
plt.show()

plt.title("Première image")
plt.imshow(data[0])
plt.show()

plt.title("Seconde image")
plt.imshow(data[1])
plt.show()

np.save('data.npy', data)

