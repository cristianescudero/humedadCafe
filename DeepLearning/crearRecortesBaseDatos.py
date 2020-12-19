import os,re,cv2,scipy.io
import numpy as np
import matplotlib.pyplot as plt
from time import time

from multiprocessing.pool import ThreadPool


dirname = os.path.join(os.getcwd(), 'D:\Bases datos\Cafe ML\con cascara')

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

tamRecortes = 500
contadorImprimirPantalla = 0
for root, dirnames, filenames in os.walk(imgpath):
    print("Se encuentra en la carpeta ----> ",root)
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            contadorImprimirPantalla = contadorImprimirPantalla + 1
            print(contadorImprimirPantalla)
            filepath = os.path.join(root, filename)
            img  = cv2.imread(filepath)                  #Lee imagen en RGB
            #print(img.shape[0],img.shape[1])

            numeroFilas = int(np.floor(img.shape[0]/tamRecortes))
            numeroColumnas = int(np.floor(img.shape[1] / tamRecortes))
            #print(filename)
            #print(numeroFilas,numeroColumnas)
            segmentos = np.zeros((tamRecortes,tamRecortes,3))
            contador = 0

            for i in range(numeroFilas-1):
                for j in range(numeroColumnas-1):
                    segmentos = img[i*tamRecortes:(i+1)*tamRecortes,j*tamRecortes:(j+1)*tamRecortes,: ]

                    contador = contador + 1
                    #cv2.imwrite('rec'+str(contador)+filename, segmentos)

                    cv2.imwrite('D:\Bases datos\Cafe DL\ ' + 'rec'+str(contador)+filename, segmentos)

                     #------------------------------------------------------
                    archivo = open("cuenta.txt", "w")
                    archivo.write('rec'+str(contador)+filename + '\n')  # limite eje

                    archivo.close()
                    #------------------------------------------------------



                    #plt.imshow(segmentos)
                    #plt.show()


