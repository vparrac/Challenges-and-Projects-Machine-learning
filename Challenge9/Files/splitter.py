from numpy import genfromtxt, newaxis, savetxt, array
from sklearn.model_selection import train_test_split
import statistics 

data = genfromtxt('clean_data.csv', delimiter=';')
labels= data[:,0][:,newaxis]
features = data[:,1:]
normFeatures=array(features)

# Normalización
for i in range(features.shape[1]):
    feature = features[:,i]
    mean = statistics.mean(feature)
    desvesta= statistics.stdev(feature)
    feature=feature-mean
    feature=feature/desvesta
    normFeatures[:,i]=feature

    
X_train_train, X_test, y_train_train, y_test = train_test_split(normFeatures, labels, test_size=0.1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_train, y_train_train, test_size=0.1)
savetxt("X_test",X_test,delimiter=";")
savetxt("y_test",y_test,delimiter=";")

savetxt("X_train",X_train,delimiter=";")
savetxt("y_train",y_train,delimiter=";")

savetxt("X_validation",X_validation,delimiter=";")
savetxt("y_validation",y_validation,delimiter=";")
