from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
modelo= load_model('problem_two.h5')
test = ImageDataGenerator().flow_from_directory('./Test_two', target_size=(720,1280), classes=['W1','W4'], batch_size=100)
print('pasa')
cmT=np.array([[0, 0],
              [0, 0]])
for i in range(47):
    imag_test, labels_test = next(test)
    print('pasa')
    y_pred = modelo.predict(imag_test)
    labelsPre=y_pred[:,0]
    for i in range(len(labelsPre)):
        if(labelsPre[i]>=0.5):
            labelsPre[i]=1
        else:
            labelsPre[i]=0
    cm=confusion_matrix(labels_test[:,0],labelsPre)
    cmT=cmT+cm
    print(cmT)