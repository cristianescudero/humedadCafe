import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scikitplot as skplt
from six import StringIO
from sklearn import tree, metrics
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

BaseDatos2=data = pd.read_csv("BaseDatosR.csv",sep=',')
Data=pd.DataFrame(BaseDatos2)
Clases=Data['clases']
del(Data['clases'])

df=Data

K=2
colores=[[(15/250,15/250,15/250)],[(150/250,150/250,150/250)],[(200/250,200/250,200/250)],[(100/250,100/250,100/200)]]    #Cada cluster sale de difrente color
#Se predicen los grupos
y_kmeans1 = KMeans(n_clusters=K)
y_kmeans =y_kmeans1.fit_predict(df)
centroids = y_kmeans1.cluster_centers_

for l in range(K):

    n=np.array(df[y_kmeans==l])                         #Se buscan cada una de las etiquetas
    total= len(n)                                       #Se obtiene el total de cada etiqueta
    print("Cluster ",l," :",total)

    Custer = np.where(np.array(y_kmeans) == l)

    Clase1 = 0
    Clase2 = 0

    for i in range(np.array(Custer).shape[1]):
        if Clases[Custer[0][i]] == 1:
            Clase1 = Clase1 + 1
        else:
            Clase2 = Clase2 + 1

    print("Clase 1: ", Clase1)
    print("Clase 2: ", Clase2)
    print("_________________________________________________")
    plt.scatter(n[:, 0], n[:, 1], s=100, c=colores[l])       # Se grafican los puntos de cada grupo de color diferente
#plt.show()

#Coeficiente de silhouette
silhouette_vals = silhouette_samples(Data, Clases)

no_of_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15]

#Coeficiente de siloueth
for n_clusters in no_of_clusters:
    cluster = KMeans(n_clusters=n_clusters)
    cluster_labels = cluster.fit_predict(df)

    silhouette_avg = silhouette_score(df, cluster_labels)

    print("For no of clusters =", n_clusters, " The average silhouette_score is :", silhouette_avg)

A=[]
#Metodo del codo
for i in range(1,20):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df)
    A.append([i,kmeans.inertia_])

plt.plot((range(1,20)),A)
plt.xlabel("Numero de clusters")
plt.ylabel("score")
plt.show()
