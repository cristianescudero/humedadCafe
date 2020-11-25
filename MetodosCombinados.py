from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scikitplot as skplt
from six import StringIO
from sklearn import tree, metrics
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier


BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
Clases=Data['clases']
del(Data['clases'])

#Partición de los datos en Xtrain, Xtest, Ytrain y Ytest
Filas=np.array(Data).shape[0]
Columnas=np.array(Data).shape[1]
Porcent1=(70*Filas)/100
Porcent2=(30*Filas)/100

rand=random.sample(range(0,Filas),Filas)

VecTrain = (rand)[:int(Porcent1)]
VecTest = np.array(rand)[int(Porcent1):]

#Se verifica que no se repitan las filas de Xtest y Xtrain
for i in range(np.array(VecTest).shape[0]):

    if VecTrain[i]==VecTest[i]:
        print("Aqui")

XTrain= np.ones((int(Porcent1),Columnas))
XTrain = Data.iloc[VecTrain]
XTest = Data.iloc[VecTest]
YTrain = Clases[VecTrain]
YTest = Clases[VecTest]

#Arbol de decision baggin
clf_tree = BaggingClassifier()
clf_tree.fit(XTrain, YTrain)

#Predicción utilizando YTrainn y YTest
YPred = clf_tree.predict(XTest)
print("Bagging")
print("")
print('Model test Score: %.3f, ' %clf_tree.score(XTest, YTest),
      'Model training Score: %.3f' %clf_tree.score(XTrain, YTrain))

print("Accuracy:",metrics.accuracy_score(YTest, YPred))
print("Precision :",metrics.precision_score(YTest, YPred))
print("Recall :",metrics.recall_score(YTest, YPred))
print("F1 :",metrics.f1_score(YTest, YPred))

#Analisis de costos manualmente
TP=0
TN=0
TPC1=0
TPC2=0
FPC1=0
FPC2=0

for i in range(np.array(YPred).shape[0]):
    if np.array(YPred)[i] == np.array(YTest)[i]:
        TP=TP+1
        if np.array(YPred)[i]==1:
            TPC1 = TPC1 + 1
        else:
            TPC2 = TPC2 + 1
    else:
        TN=TN+1
        if np.array(YPred)[i]==1:
            FPC1 = FPC1 + 1
        else:
            FPC2 = FPC2 + 1

Accuracy=TP/(TN+TP)
Precision=TPC1/(TPC1+FPC1)
Recall=TPC1/(TPC1+FPC2)
F1=2*((Precision*Recall)/(Precision+Recall))

print("Accuracy:",Accuracy)
print("Precision :", Precision)
print("Recall :", Recall)
print("F1 :", F1)

#Intervalos de confianza del metodo
c=0.95
N=np.array(YPred).shape[0]
f=TP/N
prob = (1 - c)/2
z = norm.ppf(1-prob)
D1 = (z * (math.sqrt((f/N)-((f**2)/N)+((z**2)/(4*(N**2))))))/(1+((z**2)/N))
D2 = (((z**2)/(2*N))+f/(1+((z**2)/N)))
Pinf=D2-D1
Psup=D2+D1
print("Intervalo de confianza [",Pinf*100,"    ",Accuracy*100,"    ",Psup*100,"]")

#Arbol de decision bosting
clf_tree = AdaBoostClassifier()
clf_tree.fit(XTrain, YTrain)

#Predicción utilizando YTrainn y YTest
YPred = clf_tree.predict(XTest)

print("_____________________________________________________________________")
print("Boosting")
print("")
print('Model test Score: %.3f, ' %clf_tree.score(XTest, YTest),
      'Model training Score: %.3f' %clf_tree.score(XTrain, YTrain))

print("Accuracy:",metrics.accuracy_score(YTest, YPred))
print("Precision :",metrics.precision_score(YTest, YPred))
print("Recall :",metrics.recall_score(YTest, YPred))
print("F1 :",metrics.f1_score(YTest, YPred))


#Analisis de costos manualmente
TP=0
TN=0
TPC1=0
TPC2=0
FPC1=0
FPC2=0

for i in range(np.array(YPred).shape[0]):
    if np.array(YPred)[i] == np.array(YTest)[i]:
        TP=TP+1
        if np.array(YPred)[i]==1:
            TPC1 = TPC1 + 1
        else:
            TPC2 = TPC2 + 1
    else:
        TN=TN+1
        if np.array(YPred)[i]==1:
            FPC1 = FPC1 + 1
        else:
            FPC2 = FPC2 + 1

Accuracy=TP/(TN+TP)
Precision=TPC1/(TPC1+FPC1)
Recall=TPC1/(TPC1+FPC2)
F1=2*((Precision*Recall)/(Precision+Recall))

print("Accuracy:",Accuracy)
print("Precision :", Precision)
print("Recall :", Recall)
print("F1 :", F1)

#Intervalos de confianza del metodo
c=0.95
N=np.array(YPred).shape[0]
f=TP/N
prob = (1 - c)/2
z = norm.ppf(1-prob)
D1 = (z * (math.sqrt((f/N)-((f**2)/N)+((z**2)/(4*(N**2))))))/(1+((z**2)/N))
D2 = (((z**2)/(2*N))+f/(1+((z**2)/N)))
Pinf=D2-D1
Psup=D2+D1
print("Intervalo de confianza [",Pinf*100,"    ",Accuracy*100,"    ",Psup*100,"]")

