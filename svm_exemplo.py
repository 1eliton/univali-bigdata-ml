
# --== wine quality ==--

# scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# towardsdatascience.com/visualizing-support-vector-machine-decision-boundary-69e7591dacea
# https://archive.ics.uci.edu/ml/datasets/wine+quality

import pandas as pd

csv = pd.concat([pd.read_csv('winequality-red.csv', sep=';'),
                 pd.read_csv('winequality-white.csv', sep=';')])

print('dados nulos?')
csv.isnull().any()

print('separando classe de features')
csv.classe = csv['quality']
csv.features = csv.drop(columns=['quality'], inplace=False)

print('aplicando minmax (normalizacao)')
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
#csv.classenormalizada = scaler.fit_transform(csv.classe.values)
csv.featuresnormalizadas = scaler.fit_transform(csv.features.values)

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(csv.featuresnormalizadas, csv.classe.values, test_size=0.3)

print('aplicando svm com kernel linear')

# x = atributos, y = classe
# trintaporcento = int(round(csv.featuresnormalizadas.shape[0] * .3,0))
# y_treino = csv.classe.sample(trintaporcento)
# x_treino = csv.featuresnormalizadas.sample(trintaporcento)


clf = SVC(kernel='linear')
clf.fit(X_train, y_train) # fit
predict_svm = clf.predict(X_test) # predict
acuracia_svm = metrics.accuracy_score(y_test, predict_svm)
#print('Acurácia global do svm com kernel linear:', acuracia_svm)

print('aplicando randomforest (sem balanceamento)')

rf = RandomForestClassifier()
rf.fit(X_train, y_train) #fit
predict_rfnb = rf.predict(X_test) #predict
acuracia_rfnb = metrics.accuracy_score(y_test, predict_rfnb)
#print('Acurácia global do randomforest:', acuracia_rf)

print('aplicando balanceamento para randomforest')
from imblearn.over_sampling import SMOTE
from collections import Counter
# balanceia os dados
csv.featuresbalanceadas, csv.classebalanceada = SMOTE(k_neighbors=4).fit_resample(csv.features, csv.classe)
# splita com base nos dados balanceados
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(csv.featuresbalanceadas, csv.classebalanceada, test_size=0.3)

print('aplicando randomforest (com balanceamento)')
rfb = RandomForestClassifier()
rfb.fit(X_train_b, y_train_b) #fit
predict_rfb = rfb.predict(X_test_b) #predict
acuracia_rfb = metrics.accuracy_score(y_test_b, predict_rfb)

print(f'Comparando acurácias\nRandomForest (sem balanceamento): {acuracia_rfnb}\nSVM: {acuracia_svm}\nRandomForest (com balanceamento): {acuracia_rfb}')

print('Encerrando')

