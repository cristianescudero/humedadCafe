#Con este script se cambian las etiquetas para probar como funciona la clasificación para diferentes clases
#el script lee un archivo. mat y guarda otro .mat con las etiquetas diferentes

import scipy.io
import numpy as np
import scipy.io


mat = scipy.io.loadmat('BaseDatos50HistWFHogSiftConCascara2Clases.mat')
print(mat.keys())
descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
descHOG = mat['DHOG']
descSIFT = mat['DSIFT']

y = np.transpose(mat['y'])
conteo = mat['conteo']
nombres = mat['nombres']




etiquetas = [];contador = 0;
for i in nombres:
    if i.find("Cascara10") != -1:
        etiqueta = 10
    elif i.find("Cascara11") != -1:
        etiqueta = 11
    elif i.find("Cascara12") != -1:
        etiqueta = 12
    elif i.find("Cascara13") != -1:
        etiqueta = 13
    elif i.find("Cascara14") != -1:
        etiqueta = 14
    elif i.find("Cascara15") != -1:
        etiqueta = 15
    elif i.find("Cascara16") != -1:
        etiqueta = 16
    elif i.find("Cascara17") != -1:
        etiqueta = 17
    elif i.find("Cascara18") != -1:
        etiqueta = 18

    elif i.find("Cascara19") != -1:
        etiqueta = 19
    elif i.find("Cascara20") != -1:
        etiqueta = 20
    elif i.find("Cascara21") != -1:
        etiqueta = 21
    elif i.find("Cascara22") != -1:
        etiqueta = 22

    elif i.find("Cascara23") != -1:
        etiqueta = 23
    elif i.find("Cascara25") != -1:
        etiqueta = 25
    elif i.find("Cascara26") != -1:
        etiqueta = 26
    elif i.find("Cascara27") != -1:
        etiqueta = 27
    elif i.find("Cascara29") != -1:
        etiqueta = 29

    else:
        etiqueta = 0

    etiquetas.append(etiqueta)

X = np.hstack([y,descHistograma,descFourier,descWavelet,descHOG,descSIFT])
print(etiquetas)

""" comento el código para no sobreescribir las bases de datos
scipy.io.savemat('BaseDatos50HistWFHogSiftConCascara18Clases.mat',{'y': etiquetas,
                'DH':descHistograma,'DF':descFourier,'DW':descWavelet,'DHOG':descHOG,
                'DSIFT':descSIFT,'conteo':conteo,'nombres':nombres})
"""



