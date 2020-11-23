#Este script nombra las imagenes con el nombre de la persona que lo etiqueto y la clase

import os,re,cv2

dirname = os.path.join(os.getcwd(), 'D:\Bases datos\Cafe\Sebastian\Con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
carpetas = os.listdir(imgpath)
print(carpetas)

#Nombres Moreno---------------------------------------------------------------------------------------------------------
indexCarpetas = -1
indexFotos = 1
for root, dirnames, filenames in os.walk(imgpath):
    #print("Se encuentra en la carpeta ----> ",root)
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):

            filepath = os.path.join(root, filename)
            carpetas[indexCarpetas]
            nombre = 'conCascara'+ str(carpetas[indexCarpetas]) + 'Moreno'+ str(indexFotos) +'.jpg'
            indexFotos = indexFotos + 1
            #print(nombre)
            img = cv2.imread(filepath)
            cv2.imwrite(nombre, img)
    indexCarpetas = indexCarpetas + 1



#Nombres Yurley---------------------------------------------------------------------------------------------------------


dirname = os.path.join(os.getcwd(), 'D:\Bases datos\Cafe\Yurley\Con cascara')

imgpath = dirname + os.sep
print("leyendo imagenes de ----> ",imgpath)
carpetas = os.listdir(imgpath)
print(carpetas)

#Nombres Moreno---------------------------------------------------------------------------------------------------------
indexCarpetas = -1
indexFotos = 1
for root, dirnames, filenames in os.walk(imgpath):
    #print("Se encuentra en la carpeta ----> ",root)
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):

            filepath = os.path.join(root, filename)
            carpetas[indexCarpetas]
            nombre = 'conCascara'+ str(carpetas[indexCarpetas]) + 'Yurley'+ str(indexFotos) +'.jpg'
            indexFotos = indexFotos + 1
            #print(nombre)
            img  = cv2.imread(filepath)
            cv2.imwrite(nombre, img)
    indexCarpetas = indexCarpetas + 1
