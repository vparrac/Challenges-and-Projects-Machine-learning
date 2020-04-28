from numpy import genfromtxt,random
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

C=[100,50,25,10,1,0.1,0.001,0.0001,0.00001]
gamma=[0.000001,0.00001,0.0001,0.001,0.01,0.1,0.5,0.8,1,5,25,50,100]


data = genfromtxt('Summary.csv', delimiter=',')
rdata=data[1:,:]

hm = random.rand(len(C), len(gamma))

for i in range(rdata.shape[0]):
    row=C.index(rdata[i][0])
    col=gamma.index(rdata[i][1]) 
    print(row,col)
    
    hm[row][col]=rdata[i][2]

print(hm)
ax = sns.heatmap(hm,linewidths=.5, xticklabels=gamma, yticklabels=C, annot=True,cmap="YlGnBu")
plt.ylabel('Valores de C')
plt.xlabel('Valores de Gamma')
plt.show()