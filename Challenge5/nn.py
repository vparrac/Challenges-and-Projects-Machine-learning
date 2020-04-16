from numpy import genfromtxt, newaxis, savetxt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix
import os
import glob
import csv
from numpy import genfromtxt
X_train= genfromtxt('Xtrain.txt', delimiter=';')
X_test= genfromtxt('Xtest.txt', delimiter=';')
y_train= genfromtxt('Ytrain.txt', delimiter=';')
y_test= genfromtxt('ytest.txt', delimiter=';')

# define the keras model
model = Sequential()
summary = open("summarySigmoid.csv", "w")
summary.write("af,loss,lr,nn,loss,aca \n")

epochNumber= 200
activationFunctions= ['sigmoid']
learningRates=[0.001]
loss=['mean_squared_error']
nn= [1]
for activation in activationFunctions:  
    for lr in learningRates:
        for l in loss:
            for n in nn:
                model.add(Dense(n, input_dim=30, activation=activation))
                model.add(Dense(1, activation=activation))                
                pathTrainLoss="./results/train/af/"+activation+"/loss/"+l+"/lr/"+"0.00001"+"/"+str(n)+"_loss.csv"
                pathTrainAccuracy="./results/train/af/"+activation+"/loss/"+l+"/lr/"+"0.00001"+"/"+str(n)+"_accuracy.csv"
                pathAccuracy="./results/validation/af/"+activation+"/loss/"+l+"/lr/"+"0.00001"+"/"+str(n)+".csv"               
                sgd = optimizers.SGD(lr=lr)
                model.compile(sgd, loss=l, metrics=['accuracy'])
                history_callback= model.fit(X_train, y_train, epochs=epochNumber, verbose=1, steps_per_epoch=50)
                accuracy_history = history_callback.history["accuracy"]
                loss_history = history_callback.history["loss"]
                predictions = model.predict_classes(X_test)
                accuracy = model.evaluate(X_test, y_test)
                print(history_callback.history["accuracy"])
                print("#####################################################")
                loss_history = history_callback.history["loss"]
                print(loss_history)
                print(activation+","+l+","+str(lr)+","+str(n)+","+str(accuracy[0])+","+str(accuracy[1]))
                summary.write(activation+","+l+","+str(lr)+","+str(n)+","+str(accuracy[0])+"\n")
                
summary.close()