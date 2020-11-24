import numpy as np
import scipy.io
import csv


mat = scipy.io.loadmat('BaseDatos100HistWFHogSiftConCascara18Clases.mat')
print(mat.keys())
descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
descHog = mat['DHOG']
descSift = mat['DSIFT']
y = np.transpose( mat['y'] )
X = np.hstack([y,descHistograma,descFourier,descWavelet,descHog,descSift])

print('X', X.shape,'y',y.shape)
print("H=",descHistograma.shape,"F=",descFourier.shape,"W=",descWavelet.shape,"HOG=",descHog.shape,'SIFT',descSift.shape)



""" comento el código para no sobreescribir las bases de datos
with open('BaseDatos100HistWFHogSiftConCascara18Clases.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for line in X:
        writer.writerow(line)

"""