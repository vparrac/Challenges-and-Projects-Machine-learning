from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')
data = genfromtxt('SummaryLoss.csv', delimiter=';')

reluData=data[1:20,:]
sigmoidData=data[21:,:]


#Binary cross entropy
lrReluLoss1=reluData[0:9,2]
nnReluLoss1=reluData[0:9,3]
lossReluLoss1=reluData[0:9,4]


#Mean squared error
lrReluLoss2=reluData[10:,2]
nnReluLoss2=reluData[10:,3]
lossReluLoss2=reluData[10:,4]

######################################################################

#Binary cross entropy
lrSigmoidLoss1=sigmoidData[0:9,2]
nnSigmoidLoss1=sigmoidData[0:9,3]
losSigmoidLoss1=sigmoidData[0:9,4]


#Mean squared error
lrSigmoidLoss2=sigmoidData[10:,2]
nnSigmoidLoss2=sigmoidData[10:,3]
lossSigmoidLoss2=sigmoidData[10:,4]


print(sigmoidData)
#Binary cross entropy
ax1.scatter(nnSigmoidLoss1,lrSigmoidLoss1 , losSigmoidLoss1, c='g', marker='o')
#Mean squared error
ax1.scatter(nnSigmoidLoss2, lrSigmoidLoss2, lossSigmoidLoss2, c='r', marker='*')

ax1.set_xlabel('Número de neuronas')
ax1.set_ylabel('Tasa de aprendizaje')
plt.show()