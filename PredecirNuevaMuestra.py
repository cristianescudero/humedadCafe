import cv2,funcionesCafe,pickle,numpy,sys
porcentajeTamano = 70
numeroBinsHistograma = 15


nuevaMuestra = str(sys.argv[1])
#nuevaMuestra = cv2.imread('conCascara25Yurley3372.jpg')
nuevaMuestra = funcionesCafe.reducirImagen(nuevaMuestra, porcentajeTamano)
descriptorNuevaMuestra = numpy.ravel(funcionesCafe.descriptorHistWavelet(nuevaMuestra,numeroBinsHistograma))
descriptorNuevaMuestra = descriptorNuevaMuestra.reshape(1,-1)

with open("modeloKNN18Clases70ConCascara.pkl", 'rb') as file:
    modelo = pickle.load(file)
    etiqueta = modelo.predict(descriptorNuevaMuestra)
    etiqueta = etiqueta + 8
    if etiqueta == 24:
        etiqueta += 1
    print(etiqueta)



"""
import os,re,cv2,scipy.io
import numpy as np
import matplotlib.pyplot as plt
from time import time
import funcionesCafe, pickle
from multiprocessing.pool import ThreadPool


dirname = os.path.join(os.getcwd(), 'D:\Bases datos\Cafe ML\con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
porcentajeTamano = 70
numeroBinsHistograma = 15
clase = 0
descriptorAcumulado = []; nombres = [];y=[]
contador = 0

for root, dirnames, filenames in os.walk(imgpath):
    print("Se encuentra en la carpeta ----> ",root)
    clase = clase + 1
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            start_time = time()
            img  = cv2.imread(filepath)                  #Lee imagen en RGB
            img = funcionesCafe.reducirImagen(img,porcentajeTamano) #disminuye el tamaño
            descriptor = np.ravel(funcionesCafe.descriptorHistWavelet(img,numeroBinsHistograma))
            descriptor = descriptor.reshape(1,-1)
            contador = contador + 1
            with open("modeloKNN18Clases70ConCascara.pkl", 'rb') as file:
                modelo = pickle.load(file)
                etiqueta = modelo.predict(descriptor)
                print(etiqueta,clase,contador,time() - start_time)
"""