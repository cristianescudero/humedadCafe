import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

mat = scipy.io.loadmat('BaseDatos50HWF.mat')

descHistograma = mat['DH']
descFourier = mat['DF']
descWavelet = mat['DW']
y = np.transpose( mat['y'] )
X = np.hstack([descHistograma,descFourier,descWavelet])
print('X', X.shape)

template = DecisionTreeClassifier()
mdl = AdaBoostClassifier(base_estimator=template,learning_rate=0.1, n_estimators=30)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Train Adaboost Classifer
model = mdl.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)

efi = metrics.accuracy_score(y_test, y_pred)
print(efi)




