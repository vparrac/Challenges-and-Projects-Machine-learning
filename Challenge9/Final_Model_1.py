from sklearn.svm import SVC
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
import pickle
filename = 'finalized_model_1.sav'
X_train = genfromtxt('X_train.csv', delimiter=';')
y_train = genfromtxt('y_train.csv', delimiter=';')
x_test = genfromtxt('x_test.csv', delimiter=';')
y_test = genfromtxt('y_test.csv', delimiter=';')

model = SVC(kernel='rbf', probability=True, C=25, gamma=0.01)
model.fit(X_train, y_train)
predictions = model.predict(x_test)
cm = confusion_matrix(y_test,predictions)
print(cm)
pickle.dump(model, open(filename, 'wb'))