#from __future__ import print_function
import keras
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from PIL import Image
import scipy.io
from PIL import Image
import cv2

#Se escribe la carpeta donde se encuentran las muestras
dirname = os.path.join(os.getcwd(), 'Clases500')
imgpath = dirname + os.sep
from time import time

print(dirname)
print("Detener")
images = []
directories = []
dircount = []
prevRoot=''
cant=0
print("leyendo imagenes de ",imgpath)

#Se recorren las clases de la base de datos
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)                    ##Se carga cada imagen
            newimg = cv2.resize(image,(100,100))            #se reduce el tamaño de la imagen  
            images.append(newimg)                           #Se concatenan todas las imagenes
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0

dircount.append(cant)
dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))
labels=[]
indice=0
#Se generan las etiquetas
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))



y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
num_classes = 4     #Numero de clases de la base de datos
batch_size = 12     #Cantidad de muestras que se van a tomar en cada entrenamiento
epochs = 10         #Cantidad de veces que se va a entrenar

# input image dimensions. Las imagenes estan en rgb y anteriormente se redujo de 500 a 100
img_rows, img_cols = 100, 100
canales=3
# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Para el proceso de validación se divide la base en train, test, con un 70% para train 30% test
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

print('Training data shape : ', x_train.shape, y_train.shape)
print('Testing data shape : ', x_test.shape, y_test.shape)

#se deja test y train con el formato que se requiere para ingresar a la red neuronal convolucional 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, canales)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, canales)
input_shape = (img_rows, img_cols, canales)

#Se normalizan las imagenes, antes estaban de 0 a 255 y ahora quedan de 0 a 1, por eso se debe castear a float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class esto se hace para cumplir el formato de entrada de la red
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


start_time = time()
#Se usa la funcion sequential para agregar cada capa de la red de forma secuencial
model = keras.Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(img_rows,img_cols,canales)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='tanh'))
model.add(Dense(units=84, activation='tanh'))
model.add(Dense(units=48, activation='tanh'))
model.add(Dense(units=12, activation='tanh'))
model.add(Dense(units=num_classes, activation = 'softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),  metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
#model.save("modelTESIS.h5")
#scipy.io.savemat('X_testTESIS.mat',{'x_test':x_test,'y_test':y_test})
print("Tiempo total: ",(time() - start_time))
