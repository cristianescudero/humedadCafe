import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats.stats as st
import plotly.express as px
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from scipy.linalg import svd
from numpy import diag
from numpy import zeros


'''
#Correlación
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

correlacion=Data.corr(method="pearson")
print(correlacion)
cmap=sns.diverging_palette(150,20,as_cmap=True)
sns.heatmap(correlacion, cmap=cmap)
plt.show()
'''
'''
#PCA
# cargamos los datos de entrada
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

# normalizamos los datos
scaler = StandardScaler()
df = Data.drop(['clases'], axis=1) #Se elimina la columna de etiquetas
scaler.fit(df)  # calculo la media para poder hacer la transformacion
X_scaled = scaler.transform(df)  # Ahora si, escalo los datos y los normalizo

# Instanciamos objeto PCA y aplicamos
pca = PCA().fit(X_scaled)        #Se implementa pca y se obtienen los componentes principales de los datos normalizados
X_pca = pca.transform(X_scaled)  # convertimos nuestros datos con las nuevas dimensiones de PCA
print(X_pca)
expl = pca.explained_variance_ratio_ #Se obtienen las varianzas de los datos

for i in range(np.array(expl).shape[0]):
    print("Con ",i," componentes se tiene el ",sum(expl[0:i])," de la varianza explicada")


# graficamos el acumulado de varianza explicada en las nuevas dimensiones
plt.figure(1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#screeplot
plt.figure(2)
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()
'''
'''
#Descompocisión de valores ingurales de manera manual.
# cargamos los datos de entrada
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

# normalizamos los datos
scaler = StandardScaler()
df = Data.drop(['clases'], axis=1) #Se elimina la columna de etiquetas
scaler.fit(df)  # calculo la media para poder hacer la transformacion
X_scaled = scaler.transform(df)  # Ahora si, escalo los datos y los normalizo

U, s, VT = svd(X_scaled)

#Se convierte sigma en las dimensiones correctas
Sigma=np.diag(s)
Sig=np.zeros((np.array(Data).shape[0],np.array(Data).shape[1]-1))
Sig[:np.array(Sigma).shape[0],:np.array(Sigma).shape[1]]=Sigma

#B =U.dot(Sig.dot(VT))  #Se obtiene de nuevo la matriz con los datos originales
#print(B)

#Transdormacion de los datos
T = U.dot(Sig)

plt.figure(1)
svd_values = np.arange(np.array(Sigma).shape[0]) + 1
plt.plot(svd_values, s, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
'''
'''
#Transformación de datos SVD con la funcion de skarn automatica
# cargamos los datos de entrada
BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
del(Data['clases'])

# normalizamos los datos
scaler = StandardScaler()
df = Data.drop(['clases'], axis=1) #Se elimina la columna de etiquetas
scaler.fit(df)  # calculo la media para poder hacer la transformacion
X_scaled = scaler.transform(df)  # Ahora si, escalo los datos y los normalizo

svd = TruncatedSVD(n_components=92)
svd.fit(X_scaled)
result = svd.transform(X_scaled)

expl = svd.explained_variance_ratio_ #Se obtienen las varianzas de los datos

for i in range(np.array(expl).shape[0]):
    print("Con ",i," componentes se tiene el ",sum(expl[0:i])," de la varianza explicada")


plt.figure(2)
svd_values = np.arange(92)
plt.plot(svd_values, svd.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')

# graficamos el acumulado de varianza explicada en las nuevas dimensiones
plt.figure(3)
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
'''