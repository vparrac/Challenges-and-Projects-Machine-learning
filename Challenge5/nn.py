from numpy import genfromtxt, newaxis, savetxt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix

X_train= genfromtxt('Xtrain.txt', delimiter=';')
X_test= genfromtxt('Xtest.txt', delimiter=';')
y_train= genfromtxt('Ytrain.txt', delimiter=';')
y_test= genfromtxt('ytest.txt', delimiter=';')

# define the keras model
model = Sequential()

epochNumber= 200
activationFunctions= ['sigmoid','relu']
learningRates=[0.0001,0.001,0.0001],
loss=['mean_squared_error','binary_crossentropy']
nn= [1,10,200,100,50]
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
                savetxt(pathTrainLoss,loss_history,delimiter=";")
                savetxt(pathTrainAccuracy,accuracy_history,delimiter=";")
                savetxt(pathAccuracy,accuracy,delimiter=";")             
                