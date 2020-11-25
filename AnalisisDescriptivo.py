import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats.stats as st
import plotly.express as px

'''
#Análisis de outliers y campos vacios

BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

NumClase1=np.array(np.where(np.array(Data['clases'])==1)).shape[1]
NumClase2=np.array(np.where(np.array(Data['clases'])==2)).shape[1]
#print(NumClase2,NumClase1)
#print(Data)

for i in Data:

    plt.figure(1)
    sns.boxplot(data=Data[i])

    Q1=Data[i].quantile(0.25)
    Q3=Data[i].quantile(0.75)
    IQR=Q3-Q1

    print("Valores atipicos y campos vacios en : ", i)
    Outliers=np.where((((Data[i]) < (Q1 - 1.5 * IQR))| ((Data[i]) > (Q3 + 1.5 * IQR))))
    A=np.array(Outliers[0][:]).shape[0]
    print(A)

    MV = np.array(np.where(np.array(Data[i].isnull()) == True)).shape[1]
    print(MV)
    plt.show()'''

'''
#Histogramas
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

for i in Data:
    plt.figure(1)
    plt.hist(Data[i],15,color="blue", ec="black")
    plt.show()
'''
'''
#Scatterplot
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

colum1=Data[['histo1','histo2','histo3','histo4']]
print(colum1)

sns.set(style='ticks', color_codes=True)
cmap=sns.diverging_palette(150,20,as_cmap=True)
sns.pairplot(colum1, kind='reg')
plt.show()
'''

#Datos estadisticos de las caracteristicas
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

for i in Data:

    DatosEstadisticos = Data[i].describe()
    print(DatosEstadisticos)
    print("Sesgo      ",st.skew(Data[i]))
    print("Curtosis       ",st.kurtosis(Data[i]))
    print("Varianza      ",Data[i].var())
    print("Moda      ", Data[i].mode())


