import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats.stats as st
import plotly.express as px

#Reducción de la base de datos
caracteristicas=['clases','histo1','histo2','histo3','histo4','histo5','histo6','histo7','histo8','histo9','histo10','histo11',
                 'histo12','histo13','histo14','histo15','histo16','histo17','histo18','histo19','histo20','histo21','histo22',
                 'histo23','histo24','histo25','histo26','histo27','histo28','histo29','histo30','histo31','histo32','histo33',
                 'histo34','histo35','histo36','histo37','histo38','histo39','histo40','hist41','histo42','histo43','histo44', 'histo45',
                 'Fourier1','Fourier2','Fourier3','Fourier4','Fourier5','Fourier6','Fourier7','Fourier8','Fourier9','Fourier10',
                 'Fourier11','Fourier12','wavelet1','wavelet2','wavelet3','wavelet4','wavelet5','wavelet6','wavelet7','wavelet8','wavelet9',
                 'wavelet10','wavelet11','wavelet12','wavelet13','wavelet14','wavelet15','wavelet16','wavelet17','wavelet18','wavelet19',
                 'wavelet20','wavelet21','wavelet22','wavelet23','wavelet24','wavelet25','wavelet26','wavelet27','wavelet28','wavelet29',
                 'wavelet30','wavelet31','wavelet32','wavelet33','wavelet34','wavelet35','wavelet36'
                 ]
BaseDatos2=data = pd.read_csv("BaseDatos.csv",header=None,sep=',')
BaseDatos2.columns=caracteristicas

Data=pd.DataFrame(BaseDatos2)
NumClase1=np.array(np.where(np.array(Data['clases'])==1)).shape[1]
NumClase2=np.array(np.where(np.array(Data['clases'])==2)).shape[1]
rand=random.sample(range(NumClase1+1,NumClase2+NumClase1),NumClase2-NumClase1)
#print(np.array(rand).shape)
#print(NumClase1,NumClase2)

Data=Data.drop(rand,axis=0)
#print(np.array(Data).shape)
#print(Data)

Data.to_csv('BaseDatosR.csv', index=False)

'''
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
    plt.show()
'''
'''
#Histogramas
BaseDatos=data = np.array(pd.read_csv("BaseDatos.csv",sep=','))
Target=np.unique(BaseDatos[:,0])
Filas=BaseDatos.shape[0]
Columnas=BaseDatos.shape[1]

for i in range(Columnas-1):
    plt.figure(i+1)
    plt.hist(BaseDatos[:,i+1],15,color="yellow", ec="black")
    plt.show()
'''
'''
#Correlación
caracteristicas=['clases','histo1','histo2','histo3','histo4','histo5','histo6','histo7','histo8','histo9','histo10','histo11',
                 'histo12','histo13','histo14','histo15','histo16','histo17','histo18','histo19','histo20','histo21','histo22',
                 'histo23','histo24','histo25','histo26','histo27','histo28','histo29','histo30','histo31','histo32','histo33',
                 'histo34','histo35','histo36','histo37','histo38','histo39','histo40','hist41','histo42','histo43','histo44', 'histo45',
                 'Fourier1','Fourier2','Fourier3','Fourier4','Fourier5','Fourier6','Fourier7','Fourier8','Fourier9','Fourier10',
                 'Fourier11','Fourier12','wavelet1','wavelet2','wavelet3','wavelet4','wavelet5','wavelet6','wavelet7','wavelet8','wavelet9',
                 'wavelet10','wavelet11','wavelet12','wavelet13','wavelet14','wavelet15','wavelet16','wavelet17','wavelet18','wavelet19',
                 'wavelet20','wavelet21','wavelet22','wavelet23','wavelet24','wavelet25','wavelet26','wavelet27','wavelet28','wavelet29',
                 'wavelet30','wavelet31','wavelet32','wavelet33','wavelet34','wavelet35','wavelet36'
                 ]
BaseDatos2=data = pd.read_csv("BaseDatos.csv",header=None,sep=',')
BaseDatos2.columns=caracteristicas
Data=pd.DataFrame(BaseDatos2)
print(Data)
correlacion=Data.corr(method="pearson")
cmap=sns.diverging_palette(150,20,as_cmap=True)
sns.heatmap(correlacion, cmap=cmap)
plt.show()
'''
'''
#Scatterplot
caracteristicas=['clases','histo1','histo2','histo3','histo4','histo5','histo6','histo7','histo8','histo9','histo10','histo11',
                 'histo12','histo13','histo14','histo15','histo16','histo17','histo18','histo19','histo20','histo21','histo22',
                 'histo23','histo24','histo25','histo26','histo27','histo28','histo29','histo30','histo31','histo32','histo33',
                 'histo34','histo35','histo36','histo37','histo38','histo39','histo40','hist41','histo42','histo43','histo44', 'histo45',
                 'Fourier1','Fourier2','Fourier3','Fourier4','Fourier5','Fourier6','Fourier7','Fourier8','Fourier9','Fourier10',
                 'Fourier11','Fourier12','wavelet1','wavelet2','wavelet3','wavelet4','wavelet5','wavelet6','wavelet7','wavelet8','wavelet9',
                 'wavelet10','wavelet11','wavelet12','wavelet13','wavelet14','wavelet15','wavelet16','wavelet17','wavelet18','wavelet19',
                 'wavelet20','wavelet21','wavelet22','wavelet23','wavelet24','wavelet25','wavelet26','wavelet27','wavelet28','wavelet29',
                 'wavelet30','wavelet31','wavelet32','wavelet33','wavelet34','wavelet35','wavelet36'
                 ]
BaseDatos2=data = pd.read_csv("BaseDatos.csv",header=None,sep=',')
BaseDatos2.columns=caracteristicas
Data=pd.DataFrame(BaseDatos2)
colum1=Data[['histo1','histo2','histo3','histo4']]

print(colum1)

sns.set(style='ticks', color_codes=True)
cmap=sns.diverging_palette(150,20,as_cmap=True)
sns.pairplot(colum1, kind='reg')
plt.show()
'''
'''
#Datos estadisticos de las caracteristicas
caracteristicas=['clases','histo1','histo2','histo3','histo4','histo5','histo6','histo7','histo8','histo9','histo10','histo11',
                 'histo12','histo13','histo14','histo15','histo16','histo17','histo18','histo19','histo20','histo21','histo22',
                 'histo23','histo24','histo25','histo26','histo27','histo28','histo29','histo30','histo31','histo32','histo33',
                 'histo34','histo35','histo36','histo37','histo38','histo39','histo40','hist41','histo42','histo43','histo44', 'histo45',
                 'Fourier1','Fourier2','Fourier3','Fourier4','Fourier5','Fourier6','Fourier7','Fourier8','Fourier9','Fourier10',
                 'Fourier11','Fourier12','wavelet1','wavelet2','wavelet3','wavelet4','wavelet5','wavelet6','wavelet7','wavelet8','wavelet9',
                 'wavelet10','wavelet11','wavelet12','wavelet13','wavelet14','wavelet15','wavelet16','wavelet17','wavelet18','wavelet19',
                 'wavelet20','wavelet21','wavelet22','wavelet23','wavelet24','wavelet25','wavelet26','wavelet27','wavelet28','wavelet29',
                 'wavelet30','wavelet31','wavelet32','wavelet33','wavelet34','wavelet35','wavelet36'
                 ]
BaseDatos2=data = pd.read_csv("BaseDatos.csv",header=None,sep=',')
BaseDatos2.columns=caracteristicas
Data=pd.DataFrame(BaseDatos2)
n=1

for i in caracteristicas:

    DatosEstadisticos = Data[i].describe()
    print(DatosEstadisticos)
    print("Sesgo      ",st.skew(Data[i]))
    print("Curtosis       ",st.kurtosis(Data[i]))
    print("Varianza      ",Data[i].var())
    print("Moda      ", Data[i].mode())

    print("missinng values")
    MV=np.where(np.array(Data[i].isnull())==True)
    print(MV)

    D=sns.boxplot(data=Data[i])
    print(D)

    plt.figure(1)
    sns.boxplot(data=Data[i])

    plt.figure(2)
    plt.hist(Data[i], 15, color="yellow", ec="black")
    plt.show()
'''



