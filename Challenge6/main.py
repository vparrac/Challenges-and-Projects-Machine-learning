import perceptron
import numpy as np
import backpropagation
from sklearn.model_selection import train_test_split
import pdb

relu = lambda x : np.maximum(0,x)
devRelu = lambda x : np.where(x <= 0, 0, 1)


data = np.genfromtxt('clean_data.csv', delimiter=';')

np.random.shuffle(data)
labels= data[:,0][:,np.newaxis]
features = data[:,1:]
topology=[features.shape[1],2,1]
activationFunctions=[relu,relu,relu]
#neuronalNetwork = perceptron.createNeuralNet([5,3,1],activationFunctions)


cost = lambda ypred, yreal : np.mean( ( ypred - yreal )**2 )
costDev = lambda ypred, yreal: ypred-yreal


nn = perceptron.createNeuralNet([30,1,1],[relu,relu,relu])
#out= backpropagation.train(nn, np.array([1,1,1,1,2]), [1], cost, costDev, devRelu, learningRate=0.05, train=True )

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
#pdb.set_trace()
for i in range(len(X_train)):
    backpropagation.train(nn, X_train[i], y_train[i], cost, costDev, devRelu, learningRate=0.05, train=True )
errores =0
for i in range(len(X_test)):
    out=backpropagation.train(nn, X_test[i], y_test[i], cost, costDev, devRelu, learningRate=0.05, train=False )
    #pdb.set_trace()
    if out[0][0]!=y_test[i][0]:
        errores=errores+1

print(errores/len(X_test))
print(len(X_test))
#predicho = train(nn,X_test,y_test,cost,5e-30, False)[0]
#print(predicho)
