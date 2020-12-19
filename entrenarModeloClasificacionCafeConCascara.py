import numpy as np
import scipy.io
from sklearn import metrics, neighbors
import pickle

mdl = neighbors.KNeighborsClassifier(n_neighbors=10)
mat = scipy.io.loadmat('Base70FinalHistWavReducidaConCascara.mat')
print(mat.keys())
X = mat['descriptor']
y = mat['y']

model = mdl.fit(X,np.ravel(y))
with open("modeloKNN18Clases70ConCascara.pkl", 'wb') as file:
    pickle.dump(model, file)









"""
#Con esta parte de código se genero la matriz X que permite entrenar el modelo de clasificación
dirname = os.path.join(os.getcwd(), 'D:\Bases datos\Cafe ML\con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
porcentajeTamano = 70
numeroBinsHistograma = 15
clase = 0
descriptorAcumulado = []; nombres = [];y=[]
contador = 1

start_time = time()
for root, dirnames, filenames in os.walk(imgpath):
    print("Se encuentra en la carpeta ----> ",root)
    clase = clase + 1
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):

            filepath = os.path.join(root, filename)
            img  = cv2.imread(filepath)                  #Lee imagen en RGB

            img = funcionesCafe.reducirImagen(img,porcentajeTamano) #disminuye el tamaño

            descriptor = np.ravel(funcionesCafe.descriptorHistWavelet(img,numeroBinsHistograma))

            descriptorAcumulado.append(descriptor)
            print(len(descriptorAcumulado),len(descriptorAcumulado[0]),clase,contador)
            contador = contador + 1
            y.append(clase)
            nombres.append(filename)

            if contador % 10 == 0:
                scipy.io.savemat('Base70FinalHistWavReducidaConCascara.mat',{'descriptor': descriptorAcumulado, 'y': y,'conteo': contador, 'nombres': nombres})
                print(contador)



elapsed_time = (time() - start_time)/60
print('tiempo (min) = ' ,elapsed_time)
"""





















"""
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

ordenCaracteristicas = np.array([ 21,73,20,85,89,77,22,19,83,71,2,87,3 ,18,65,55,59,12,4,6,51,17,75,27,61,28,13,
                                  92,80,68,37,63,23,5,30,60,47,38,49,36,50,15,35,76,58,78,88,32,95,96,108,99,11,90,64,
                                  57,72,8,100,84,53,33,9,34,107,43,7,94,70,104,82,25,48,62,54,103,98,31,44,97,39,45,
                                  14,40,66,10,26,29,16,74,46,41,86,56,106,102,105,101,52,24,42,93,81,69,67,79,91])
"""






