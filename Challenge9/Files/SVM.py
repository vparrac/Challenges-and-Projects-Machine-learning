from sklearn.svm import SVC
from numpy import genfromtxt
C=[100,50,25,10,1,0.1,0.001,0.001,0.0001,0.00001]
gamma=[0.001,0.01,0.1,0.5,1,5,25,50,100]

X_train = genfromtxt('X_train', delimiter=';')
y_train = genfromtxt('y_train', delimiter=';')
x_validation = genfromtxt('X_validation', delimiter=';')
y_validation = genfromtxt('y_validation', delimiter=';')


summary = open("Summary.csv", "w")
summary.write("c,gamma,aca \n")

for c in C:
    for g in gamma:
        model = SVC(kernel='rbf', probability=True, C=c, gamma=g)
        model.fit(X_train, y_train)
        predictions = model.predict(x_validation)
        aca = model.score(x_validation,y_validation)
        summary.write(str(c)+","+str(g)+","+str(aca)+"\n")
        print(aca)
                       
summary.close()