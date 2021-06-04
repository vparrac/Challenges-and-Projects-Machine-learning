from numpy import genfromtxt
from collections import Counter


y_validation = genfromtxt('y_test', delimiter=';')
y_test = genfromtxt('y_validation', delimiter=';')
y_train = genfromtxt('y_train', delimiter=';')

counter_validation = Counter(y_validation)
counter_test = Counter(y_test)
counter_train = Counter(y_train)
print('Validation')
print(counter_validation)
print('Test')
print(counter_test)
print('Train')
print(counter_train)

