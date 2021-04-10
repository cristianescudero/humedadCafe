# humedadCafe
El orden para ejecutar los códigos es el siguiente

Los archivos NombrarImagenes.py cambian el nombre original, por un nombre que contiene si es con cascara o sin cascara, el nombre de la persona que los generó
y el número de imagen al que corresponda


Despues se debe realizar la extracción de características con el archivo extracciónCaracteristicasHumedadCafe.py, este código  obtiene los descriptores que más información
generaron, esto lo hace usando hilos para que sea más rápido el proceso y se aproveche mejor la workstation, además llama al archivo funcionescafe, donde se encuentran
todas las funciones necesarias para extraer dichos descriptores.


Este paso es opcional, el archivo curvaEficienciaVsNumCaracteristicasGenerarDatos.py hace un proceso de validación cruzada para evaluar la eficiencia vs el numero de
características extraidas, esto se hace con el fin de revisar cuantas característiicas se requieren para obtener una eficiencia de seada.


El entrenamiento del modelo se realiza con el archivo entrenarModeloClasificacionCafeConCascara.py, en este se carga la base de datos que se generó en la extracción de descriptores
y entrena un modelo knn, que fue el mejor modelo entre las pruebas realizadas.

El archivo validación cruzada evalua los diferentes modelos y genera los resultados de eficiencia con la relevancia estadística necesaria.

Finalmente PredecirNuevaMuestra.py se utiliza para predecir nuevas muestras usando el modelo entrenado.
