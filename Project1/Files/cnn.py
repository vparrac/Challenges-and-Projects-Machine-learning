import numpy as np
np.random.seed(2)
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from datetime import datetime

incioCarga = datetime.now()

train_path = './data/video1/train_problem_two'
test_path = './data/video1/test_problem_two'

train_batch = ImageDataGenerator().flow_from_directory(train_path, target_size=(720,1280), classes=['W1','W4'], batch_size=10)
test_batch = ImageDataGenerator().flow_from_directory(test_path, target_size=(720,1280), classes=['W1','W4'], batch_size=4)

finCarga = datetime.now()
#train_path = './data/video1/train_problem_one'
#test_path = './data/video1/test_problem_one'

#train_batch = ImageDataGenerator().flow_from_directory(train_path, target_size=(720,1280), classes=['W2','W6'], batch_size=10)
#test_batch = ImageDataGenerator().flow_from_directory(test_path, target_size=(720,1280), classes=['W2','W6'], batch_size=4)

#(imgs,labels)=next(train_batch)

#print(imgs[0])

#Métodos tomados y modificados de https://www.youtube.com/watch?v=LhEMXbjGV_4
def plots(ims, figsize=(25,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1]!=3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()


def plotsTwo(ims, figsize=(25,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1]!=3):
            ims = ims.transpose((0,2,3,1))    
    for i in range(len(ims)):        
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        plt.show()

modelo = Sequential()
modelo.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(720,1280,3)))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(32, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Flatten())

modelo.add(Dense(75, activation='relu'))
#modelo.add(Dense(35, activation='relu'))

modelo.add(Dense(2, activation='softmax'))

modelo.summary()

modelo.compile(SGD(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
inicioFit = datetime.now()
modelo.fit_generator(train_batch, steps_per_epoch=7,validation_data=test_batch,validation_steps=4,epochs=11,verbose=1)
finFit = datetime.now()
modelo.save('problem_two_two.h5')

print('Tiempo total')
print(finFit-incioCarga)
print('Tiempo carga')
print(finFit-incioCarga)
print('Tiempo ajuste')
print(finFit-inicioFit)
