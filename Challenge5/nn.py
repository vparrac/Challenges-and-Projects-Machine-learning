from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense

data = genfromtxt('clean_data.csv', delimiter=';')
labels= data[:,0][:,np.newaxis]
features = data[:,1:]