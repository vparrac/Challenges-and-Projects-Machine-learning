from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
data = np.genfromtxt('clean_data.csv', delimiter=';')
labels= data[:,0][:,np.newaxis]
features = data[:,1:]