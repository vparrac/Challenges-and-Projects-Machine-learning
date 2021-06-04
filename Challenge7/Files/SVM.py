from sklearn.svm import SVC
from numpy import genfromtxt

C=[1,0.1,0.001,0.001,0.0001,0.00001]
degrees=[1,2,3,4,5,6]
gamma=['scale','auto']

X_train = genfromtxt('X_train', delimiter=';')
y_train = genfromtxt('y_train', delimiter=';')
x_validation = genfromtxt('X_validation', delimiter=';')
y_validation = genfromtxt('y_validation', delimiter=';')


summary = open("Summary.csv", "w")
summary.write("c,degree,gamma,aca \n")

for c in C:
    for d in degrees:
        for g in gamma:
            model = SVC(kernel='poly', probability=True, C=c,degree=d, gamma=g)
            model.fit(X_train, y_train)
            predictions = model.predict(x_validation)
            aca = model.score(x_validation,y_validation)
            summary.write(str(c)+","+str(d)+","+g+","+str(aca)+"\n")
            print(aca)
                       
summary.close()