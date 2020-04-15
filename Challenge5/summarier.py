import os
import glob
import csv
from numpy import genfromtxt

path = './results/validation'
summary = open("summary.csv", "a")
summary.write("af,loss,lr,nn,loss \n")


files = [f for f in glob.glob(path + "**/**/*.csv", recursive=True)]

for f in files:
    document=f.split('\\')
    idxaf=document.index("af")+1
    idxlss=document.index("loss")+1
    idxlr=document.index("lr")+1
    idxnn=len(document)-1
    nndot=document[idxnn].split(".")
    nn=nndot[0]
    f2 = genfromtxt(f,delimiter=',')[0]
    st=document[idxaf]+","+document[idxlss]+","+document[idxlr]+","+nn+","+str(f2)+"\n"    
    summary.write(st)


summary.close()  

    
