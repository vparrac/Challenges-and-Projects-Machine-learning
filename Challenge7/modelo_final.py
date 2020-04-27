import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt

filename = 'finalized_model.sav'

X_train = genfromtxt('X_train', delimiter=';')
y_train = genfromtxt('y_train', delimiter=';')
x_test = genfromtxt('X_test', delimiter=';')
y_test = genfromtxt('y_test', delimiter=';')


model = SVC(kernel='poly', probability=True, C=10,degree=3)
model.fit(X_train, y_train)
predictions = model.predict(x_test)
aca = model.score(x_test,y_test)

cm = confusion_matrix(y_test,predictions)

print(cm)
print(aca)
pickle.dump(model, open(filename, 'wb'))