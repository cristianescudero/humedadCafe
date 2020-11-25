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
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus


#Arbol de decisión
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

#Arbol de decision
clf_tree = DecisionTreeClassifier()
clf_tree.fit(XTrain, YTrain)
'''
#Diagrama 1 del arbol de decisión
fig, ax = plt.subplots()
tree.plot_tree(clf_tree)
plt.show()
'''
#Predicción utilizando YTrainn y YTest
YPred = clf_tree.predict(XTest)

#Analisis de costos por medio de la funcion metrics
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

#Curva ROC
fpr,tpr, t =metrics.roc_curve(YTest,YPred, pos_label=1)
auc=metrics.roc_auc_score(YTest,YPred)
plt.plot(tpr,fpr, label="data 1 + auc"+str(auc))
plt.legend(loc=4)
plt.show()

#Curva ROC 2
skplt.metrics.plot_roc_curve(YTest,YPred)
plt.show()

#Curva lift
skplt.metrics.plot_cumulative_gain(YTest,YPred)
plt.show()


'''
#ArbolDesicionImagen
feature_cols=[   'histo1','histo2','histo3','histo4','histo5','histo6','histo7','histo8','histo9','histo10','histo11',
                 'histo12','histo13','histo14','histo15','histo16','histo17','histo18','histo19','histo20','histo21','histo22',
                 'histo23','histo24','histo25','histo26','histo27','histo28','histo29','histo30','histo31','histo32','histo33',
                 'histo34','histo35','histo36','histo37','histo38','histo39','histo40','hist41','histo42','histo43','histo44', 'histo45',
                 'Fourier1','Fourier2','Fourier3','Fourier4','Fourier5','Fourier6','Fourier7','Fourier8','Fourier9','Fourier10',
                 'Fourier11','Fourier12','wavelet1','wavelet2','wavelet3','wavelet4','wavelet5','wavelet6','wavelet7','wavelet8','wavelet9',
                 'wavelet10','wavelet11','wavelet12','wavelet13','wavelet14','wavelet15','wavelet16','wavelet17','wavelet18','wavelet19',
                 'wavelet20','wavelet21','wavelet22','wavelet23','wavelet24','wavelet25','wavelet26','wavelet27','wavelet28','wavelet29',
                 'wavelet30','wavelet31','wavelet32','wavelet33','wavelet34','wavelet35','wavelet36'
                 ]
dot_data = StringIO()
export_graphviz(clf_tree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['1','2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('ArbolData.png')
Image(graph.create_png())
'''