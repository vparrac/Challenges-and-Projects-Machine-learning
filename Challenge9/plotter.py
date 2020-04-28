from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')
data = genfromtxt('Summary_Final.csv', delimiter=',')
print(data)
c=data[:,0]
degree=data[:,1]
aca=data[:,3]



ax1.scatter(c, degree, aca, c='g', marker='*')

ax1.set_xlabel('C')
ax1.set_ylabel('Grado Polinomio')
plt.show()