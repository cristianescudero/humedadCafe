import numpy as np
import scipy.io
import csv


mat = scipy.io.loadmat('BaseDatosHFW.mat')
descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
y = np.transpose( mat['y'] )
X = np.hstack([y,descHistograma,descFourier,descWavelet])

print('X', X.shape,'y',y.shape)
print("H=",descHistograma.shape,"F=",descFourier.shape,"W=",descWavelet.shape)




with open('BaseDatosHFW.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for line in X:
        writer.writerow(line)

