import scipy.io
import numpy as np
import scipy.io


mat = scipy.io.loadmat('BaseDatos70HistWFHog.mat')
print(mat.keys())
descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
descHOG = mat['DHOG']

y = np.transpose( mat['y'] )
conteo = mat['conteo']
nombres = mat['nombres']




matSIFT = scipy.io.loadmat('SoloSIFT70.mat')
descSIFT = matSIFT['DSIFT']


#X = np.hstack([y,descHistograma,descFourier,descWavelet])


scipy.io.savemat('BaseDatos70HistWFHogSift.mat',
                 {'DSIFT': descSIFT, 'DHOG': descHOG, 'DH': descHistograma,
                  'DF': descFourier, 'DW': descWavelet, 'y': y, 'conteo': conteo, 'nombres': nombres})


"""mat = scipy.io.loadmat('BaseDatosHFW.mat')
descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
y = np.transpose( mat['y'] )
X = np.hstack([y,descHistograma,descFourier,descWavelet])

print('X', X.shape,'y',y.shape)
print("H=",descHistograma.shape,"F=",descFourier.shape,"W=",descWavelet.shape)

scipy.io.savemat('BaseDatos50HistWFHogSift.mat',
                 {'DSIFT': descSIFTAcumulado, 'DHOG': descHOGAcumulado, 'DH': descHistAcumulado,
                  'DF': descFourAcumulado, 'DW': descWaveAcumulado, 'y': y, 'conteo': conteo, 'nombres': nombres})
"""