import numpy as np

matrizConfusion = [[1226, 184],[115, 3572]]

print("Eficiencia Arbol= ",94.1,'%')
print("Matriz confusion muestras")
for row in matrizConfusion:
    for elem in row:
        print(elem, end=' ')
    print()
print("Matriz confusion porcentaje")

matrizConfusion = [[87, 13],[3, 97]]

for row in matrizConfusion:
    for elem in row:
        print(elem,'%', end=' ')
    print()