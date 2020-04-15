from numpy import genfromtxt, newaxis, savetxt
from sklearn.model_selection import train_test_split

data = genfromtxt('clean_data.csv', delimiter=';')
labels= data[:,0][:,newaxis]
features = data[:,1:]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
savetxt("Xtrain",X_train,delimiter=";")
savetxt("Ytrain",y_train,delimiter=";")
savetxt("Xtest",X_test,delimiter=";")
savetxt("ytest",y_test,delimiter=";")

