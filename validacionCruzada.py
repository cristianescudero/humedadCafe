import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics, svm,neighbors
import time


#Parámetros validación cruzada---------------------------------------------------------------------
kfolds = 100


#Cargar datos--------------------------------------------------------------------------------------
mat = scipy.io.loadmat('../../X.mat')
descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
descHOG = mat['DHOG']
descSIFT = mat['DSIFT']
y = np.transpose( mat['y'] )
X = np.hstack([descHistograma,descFourier,descWavelet,descHOG,descSIFT])
print('X', X.shape)


#Variables ------------------------------------------------------------------------------------------
cantidadClases = len(np.unique(y));
eficienciaArbolEnsamblado = [];eficienciaSVM = [];eficienciaKNN = []; eficienciaArbolSimple = [];
matrizConfusion1 = np.zeros((cantidadClases,cantidadClases,kfolds))
matrizConfusion2 = np.zeros((cantidadClases,cantidadClases,kfolds))
matrizConfusion3 = np.zeros((cantidadClases,cantidadClases,kfolds))
matrizConfusion4 = np.zeros((cantidadClases,cantidadClases,kfolds))


arbolEnsambladoMediaMatriz = np.zeros((cantidadClases,cantidadClases));arbolEnsambladoStdMatriz = np.zeros((cantidadClases,cantidadClases));
svmMediaMatriz = np.zeros((cantidadClases,cantidadClases)); svmSTDMatriz = np.zeros((cantidadClases,cantidadClases));
knnMediaMatriz = np.zeros((cantidadClases,cantidadClases)); knnSTDMatriz = np.zeros((cantidadClases,cantidadClases));
arbolSimpleMediaMatriz = np.zeros((cantidadClases,cantidadClases)); arbolSimpleSTDMatriz = np.zeros((cantidadClases,cantidadClases));





# Modelos computados ------------------------------------------------------------------------------
templateDecisionTree = DecisionTreeClassifier()
mdl1 = AdaBoostClassifier(base_estimator=templateDecisionTree,learning_rate=0.1, n_estimators=30)
mdl2 = svm.SVC(kernel='poly')
mdl3 = neighbors.KNeighborsClassifier(n_neighbors=10)
mdl4 = DecisionTreeClassifier()

#-------------------------------------------------------------------------------------------------
start_time = time.time()


for i in range(kfolds):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train Adaboost Classifer
    model1 = mdl1.fit(X_train, y_train.ravel())
    model2 = mdl2.fit(X_train, y_train.ravel())
    model3 = mdl3.fit(X_train, y_train.ravel())
    model4 = mdl4.fit(X_train, y_train.ravel())

    #Predict the response for test dataset
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)
    y_pred4 = model4.predict(X_test)


    eficienciaArbolEnsamblado.append(metrics.accuracy_score(y_test,y_pred1)*100)
    eficienciaSVM.append(metrics.accuracy_score(y_test,y_pred2)*100)
    eficienciaKNN.append(metrics.accuracy_score(y_test,y_pred3)*100)
    eficienciaArbolSimple.append(metrics.accuracy_score(y_test,y_pred4)*100)

    matrizConfusion1[:, :, i] = metrics.confusion_matrix(y_test, y_pred1)
    matrizConfusion2[:, :, i] = metrics.confusion_matrix(y_test, y_pred2)
    matrizConfusion3[:, :, i] = metrics.confusion_matrix(y_test, y_pred3)
    matrizConfusion4[:, :, i] = metrics.confusion_matrix(y_test, y_pred4)



#Lo convierto en porcentaje
for k in range(kfolds):
    for i in range(cantidadClases):
        sumaClase1 = np.sum(matrizConfusion1[i, :, k])
        sumaClase2 = np.sum(matrizConfusion2[i, :, k])
        sumaClase3 = np.sum(matrizConfusion3[i, :, k])
        sumaClase4 = np.sum(matrizConfusion4[i, :, k])
        for j in range(cantidadClases):
            matrizConfusion1[i, j, k] = (matrizConfusion1[i, j, k] / (sumaClase1)) *100

            matrizConfusion2[i, j, k] = (matrizConfusion2[i, j, k] / (sumaClase2)) * 100
            matrizConfusion3[i, j, k] = (matrizConfusion3[i, j, k] / (sumaClase3)) * 100
            matrizConfusion4[i, j, k] = (matrizConfusion4[i, j, k] / (sumaClase4)) * 100

#Calculo la media y desviación estandar
for i in range(cantidadClases):
    for j in range(cantidadClases):
        arbolEnsambladoMediaMatriz[i,j] = np.mean(matrizConfusion1[i,j,:])
        arbolEnsambladoStdMatriz[i,j] = np.std(matrizConfusion1[i,j,:])


        svmMediaMatriz[i, j] = np.mean(matrizConfusion2[i, j, :])
        svmSTDMatriz[i, j] = np.std(matrizConfusion2[i, j, :])

        knnMediaMatriz[i, j] = np.mean(matrizConfusion3[i, j, :])
        knnSTDMatriz[i, j] = np.std(matrizConfusion3[i, j, :])

        arbolSimpleMediaMatriz[i,j] = np.mean(matrizConfusion4[i, j, :])
        arbolSimpleSTDMatriz[i,j] = np.std(matrizConfusion4[i, j, :])


print("Arbol ensamblado = ",np.mean(eficienciaArbolEnsamblado),np.std(eficienciaArbolEnsamblado))
print("SVM = ",np.mean(eficienciaSVM),np.std(eficienciaSVM))
print("KNN = ",np.mean(eficienciaKNN),np.std(eficienciaKNN))
print("Arbol simple = ",np.mean(eficienciaArbolSimple),np.std(eficienciaArbolSimple))
print("--- %s seconds ---" % (time.time() - start_time))

arbolEnsamblado = {
    "media" : np.mean(eficienciaArbolEnsamblado),
    "std" : np.std(eficienciaArbolEnsamblado),
    "cmMedia": arbolEnsambladoMediaMatriz,
    "cmStd": arbolEnsambladoStdMatriz,
    "matrices":matrizConfusion1
}

svm = {
    "media" : np.mean(eficienciaSVM),
    "std" : np.std(eficienciaSVM),
    "cmMedia": svmMediaMatriz,
    "cmStd": svmSTDMatriz,
    "matrices":matrizConfusion2
}
knn = {
    "media" : np.mean(eficienciaKNN),
    "std" : np.std(eficienciaKNN),
    "cmMedia": knnMediaMatriz,
    "cmStd": knnSTDMatriz,
    "matrices":matrizConfusion3
}


arbol = {
    "media" : np.mean(eficienciaArbolSimple),
    "std" : np.std(eficienciaArbolSimple),
    "cmMedia": arbolSimpleMediaMatriz,
    "cmStd": arbolSimpleSTDMatriz,
    "matrices":matrizConfusion4
}

scipy.io.savemat('../../datosBorrar.mat', {'arbolEnsamblado':arbolEnsamblado, 'svm':svm, 'knn':knn, 'arbol':arbol})

