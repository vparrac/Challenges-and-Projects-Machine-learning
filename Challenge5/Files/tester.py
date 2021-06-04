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

model = Sequential()
model.add(Dense(200, input_dim=30, activation='exponential'))
model.add(Dense(1, activation='exponential'))
sgd = optimizers.SGD(lr=1)
model.compile(sgd, loss='mean_squared_error', metrics=['accuracy'])
history_callback= model.fit(X_train, y_train, epochs=10, verbose=1, steps_per_epoch=50)
accuracy_history = history_callback.history["accuracy"]
loss_history = history_callback.history["loss"]
savetxt("pruebaLoss.csv",accuracy_history,delimiter=";")
savetxt("pruebaAccuracy.csv",loss_history,delimiter=";")
accuracy = model.evaluate(X_test, y_test)[1]
print(accuracy)