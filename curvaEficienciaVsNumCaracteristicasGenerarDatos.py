import csv
import numpy as np
import scipy.io
import matplotlib.pyplot as plt




from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics, svm,neighbors
import time


#este script evalua la eficiencia respecto al numero de características, a partir de la métrica
#generada por Yurley, el orden de la base de datos se encuentra en escudeero/BasesDatsGeneradas/Caracteristicas Relevantes
mat = scipy.io.loadmat('BasesDatosGeneradas/completas/BaseDatos70HistWFHogSiftConCascara18Clases.mat')

print(mat.keys())

descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
descHog = mat['DHOG']
descSift = mat['DSIFT']
y = np.transpose( mat['y'] )
X = np.hstack([y,descHistograma,descFourier,descWavelet,descHog,descSift])

kfolds = 3

cantidadClases = len(np.unique(y));
"""
ordenCaracteristicas = np.array([95,96,31,89,16,108,77,6,30,65,10,60,28,19,29,72,20,48,107,3,49,21,26,37,22,4,50,64,84,
                                 104,5,12,76,36,7,9,24,23,38,70,53,94,57,45,58,88,35,82,103,2,54,18,46,25,52,78,90,39,
                                 74,41,14,27,99,61,44,102,8,101,59,86,75,32,42,98,11,97,80,63,87,17,62,92,100,73,68,15,
                                 13,34,85,106,105,83,71,47,56,66,43,55,40,51,67,91,79,81,93,33,69])"""

ordenCaracteristicas = np.array([ 21,73,20,85,89,77,22,19,83,71,2,87,3 ,18,65,55,59,12,4,6,51,17,75,27,61,28,13,
                                  92,80,68,37,63,23,5,30,60,47,38,49,36,50,15,35,76,58,78,88,32,95,96,108,99,11,90,64,
                                  57,72,8,100,84,53,33,9,34,107,43,7,94,70,104,82,25,48,62,54,103,98,31,44,97,39,45,
                                  14,40,66,10,26,29,16,74,46,41,86,56,106,102,105,101,52,24,42,93,81,69,67,79,91])


ordenCaracteristicas = ordenCaracteristicas - 1
xOrdenada = X[:,ordenCaracteristicas]

x = range(0,xOrdenada.shape[1])


eficienciaArbolSimple = []
eficienciaArbolSimpleTotal = np.zeros((xOrdenada.shape[1],2))
mdl4 = DecisionTreeClassifier()
start_time = time.time()

for i in range(1,xOrdenada.shape[1]):
    xOrdenadaActual =  xOrdenada[:,0:i]
    print(i,"/",xOrdenada.shape[1])
    for j in range(kfolds):
        X_train, X_test, y_train, y_test = train_test_split(xOrdenadaActual, y, test_size=0.3)
        # Train Adaboost Classifer
        model4 = mdl4.fit(X_train, y_train.ravel())

        #Predict the response for test dataset

        y_pred4 = model4.predict(X_test)


        eficienciaArbolSimple.append(metrics.accuracy_score(y_test,y_pred4)*100)
        #print(eficienciaArbolSimple)
    #print(np.mean(eficienciaArbolSimple))
    eficienciaArbolSimpleTotal[i,0] = np.mean(eficienciaArbolSimple)
    eficienciaArbolSimpleTotal[i,1] = np.std(eficienciaArbolSimple)
    eficienciaArbolSimple = [];
print(eficienciaArbolSimpleTotal)

#scipy.io.savemat('curvaEficienciaVsNumCaracteristicas.mat',{'eficienciaMedia':eficienciaArbolSimpleTotal[:,0],'eficienciaStd':eficienciaArbolSimpleTotal[:,1]})


plt.errorbar(x,eficienciaArbolSimpleTotal[:,0],eficienciaArbolSimpleTotal[:,1]*1)
plt.show()





