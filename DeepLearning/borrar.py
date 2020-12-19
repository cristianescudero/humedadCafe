import numpy as np


lista = np.zeros((6,6))
lista[2,3]= 1;lista[3,5]= 1;lista[4,2]= 1;

print(lista)

fil,col = np.where(lista==1)
print(np.max(fil),np.max(col))

print(type(lista))