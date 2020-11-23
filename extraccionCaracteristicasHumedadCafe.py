import os,re,cv2,scipy.io
import numpy as np
import matplotlib.pyplot as plt
from time import time
import funcionesCafe
from multiprocessing.pool import ThreadPool


dirname = os.path.join(os.getcwd(), 'D:\Bases datos\CafeProcesado\con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
porcentajeTamano = 50
numeroBinsHistograma = 15
hiloW = ThreadPool(processes=1)
hiloF = ThreadPool(processes=1)
hiloH = ThreadPool(processes=1)
hiloHOG1 = ThreadPool(processes=1)
hiloSIFT1 = ThreadPool(processes=1)

descSIFTAcumulado = [];descHOGAcumulado = [];descHistAcumulado = [];descFourAcumulado = [];descWaveAcumulado = [];y = [];clase = -1 ; conteo =[];contador = 0;nombres =[];
start_time = time()
for root, dirnames, filenames in os.walk(imgpath):
    print("Se encuentra en la carpeta ----> ",root)
    clase = clase + 1
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):

            filepath = os.path.join(root, filename)
            img  = cv2.imread(filepath)                  #Lee imagen en RGB

            img = funcionesCafe.reducirImagen(img,porcentajeTamano) #disminuye el tamaño


            hiloWavelet = hiloW.apply_async(funcionesCafe.wavelet, args=(img, 0))
            hiloFourier = hiloF.apply_async(funcionesCafe.fourier, args=(img, 0))
            hiloHistograma = hiloH.apply_async(funcionesCafe.histograma,args=(img,numeroBinsHistograma))
            hiloHOG = hiloHOG1.apply_async(funcionesCafe.hogDescriptor,args=(img, 0))
            hiloSIFT = hiloSIFT1.apply_async(funcionesCafe.siftDescriptor,args=(img, 0))

            descHistActual = hiloHistograma.get()
            descFourActual = hiloFourier.get()
            descSiftActual = hiloSIFT.get()
            descWaveActual = hiloWavelet.get()
            descHogActual = hiloHOG.get()

            descFourActual = np.ravel(descFourActual)
            descWaveActual = np.ravel(descWaveActual)
            descSiftActual = np.ravel(descSiftActual)

            descHistAcumulado.append(descHistActual)
            descFourAcumulado.append(descFourActual)
            descWaveAcumulado.append(descWaveActual)
            descHOGAcumulado.append(descHogActual)
            descSIFTAcumulado.append(descSiftActual)

            contador = contador + 1
            conteo.append(contador)
            y.append(clase)
            nombres.append(filename)



            if contador % 10 == 0:
                scipy.io.savemat('BaseDatos50HistWFHogSift.mat',{'DSIFT': descSIFTAcumulado,'DHOG': descHOGAcumulado,'DH': descHistAcumulado, 'DF': descFourAcumulado, 'DW': descWaveAcumulado, 'y': y,'conteo': conteo, 'nombres': nombres})
                print(contador)

scipy.io.savemat('BaseDatos50HistWFHogSift.mat',{'DSIFT': descSIFTAcumulado,'DHOG': descHOGAcumulado,'DH': descHistAcumulado, 'DF': descFourAcumulado, 'DW': descWaveAcumulado, 'y': y,'conteo': conteo, 'nombres': nombres})
elapsed_time = (time() - start_time)/60
print('tiempo (min) = ' ,elapsed_time)


