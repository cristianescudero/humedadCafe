import os,re,cv2,scipy.io,scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from time import time
import funcionesCafe
from multiprocessing.pool import ThreadPool

#este script se utilizó para calcular la primera prueba de hog con la base de datos completa

dirname = os.path.join(os.getcwd(), 'D:\Bases datos\CafeProcesado\con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
porcentajeTamano = 50
numeroBinsHistograma = 15


descHistAcumulado = [];descFourAcumulado = [];descWaveAcumulado = [];y = [];clase = -1 ; conteo =[];contador = 0;nombres =[];
start_time = time()
descHOGAcumulado = [];
for root, dirnames, filenames in os.walk(imgpath):
    print("Se encuentra en la carpeta ----> ",root)
    clase = clase + 1
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):

            filepath = os.path.join(root, filename)
            img  = cv2.imread(filepath)                  #Lee imagen en RGB
            #img = funcionesCafe.reducirImagen(img,50)
            start_time = time()

            descHOGAcumulado.append(funcionesCafe.hogDescriptor(img,0))




            contador = contador + 1
            conteo.append(contador)
            y.append(clase)
            nombres.append(filename)
            scipy.io.savemat('SoloHOG.mat', {'DHOG': descHOGAcumulado, 'y': y, 'conteo': conteo, 'nombres': nombres})

            if contador % 10 == 0:
                #scipy.io.savemat('BaseDatos50HWF.mat',{'DH': descHistAcumulado, 'DF': descFourAcumulado, 'DW': descWaveAcumulado, 'y': y,'conteo': conteo, 'nombres': nombres})
                print(contador)


#scipy.io.savemat('BaseDatos50HWF.mat',{'DH':descHistAcumulado,'DF':descFourAcumulado,'DW':descWaveAcumulado,'y':y,'conteo':conteo,'nombres':nombres})
elapsed_time = (time() - start_time)/60
print('tiempo (min) = ' ,elapsed_time)


