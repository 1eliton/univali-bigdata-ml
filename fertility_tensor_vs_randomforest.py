
####
# Aula 6 - dataset fertility em modelo de tensores (DNN) e comparação com modelo de classificação RandomForest
####

###
### --== tensores ==--
###

from pickle import dump
import random
import numpy
import pandas as pd
from category_encoders import one_hot
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

csv = pd.read_csv('fertility_Diagnosis.csv', sep=',')

#print('tem dados nulos no dataset?')
#csv.isnull().any()

print('separando classe de features')
csv.classe = csv['Output']
csv.features = csv.drop(columns=['Output'], inplace=False)

print('aplicando minmax (normalizacao)')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# TODO duvida: a classe eu tb normalizo?
categoricas = one_hot.OneHotEncoder(cols=['Output'], use_cat_names=True).fit_transform(csv.classe)
csv.classenormalizada = categoricas[['Output_N', 'Output_O']].values
csv.featuresnormalizadas = csv.features.values

print('separando datasets de teste e de treino')
x_train, x_test, y_train, y_test = train_test_split(csv.featuresnormalizadas, csv.classenormalizada, test_size=0.3)

print('preparando parametros para a rede neural')
# cria as camadas da rede neural
nodes_h1_1 = 20
nodes_h1_2 = 20
# determina o n. de classes distintas
n_classes = 2
# n de features do dataset
n_features = csv.featuresnormalizadas.shape[1]
# configura execucao
epochs = 100
batch_size = 4 # n de instancias processadas por epoca

# configura a rede neural
import tflearn
net = tflearn.input_data(shape=[None, n_features])
net = tflearn.fully_connected(net, nodes_h1_1)
net = tflearn.fully_connected(net, nodes_h1_2)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# cria o modelo e configura o caminho do tensorflow
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# treino
print('treinando (fit)....')
fit = model.fit(x_train, y_train, n_epoch=epochs, batch_size=batch_size, show_metric=True)

# classificar 2 instacias qualquer
print('classificando aleatorios')
for i in range(0, 1):
    n = random.randint(0, x_test.shape[0] -1)
    amostra = x_test[n]
    randompredict = model.predict([amostra])
    classepredita = 'N' if randompredict[0][0] > randompredict[0][1] else '0'
    print('Para a amostra', amostra, ': % para N: ', randompredict[0][0], '. % para O: ', randompredict[0][1])

print('medindo a acurácia')
acuracia = model.evaluate(x_test, y_test)
print('acuracia do modelo DNN [rede neural profunda] é de ', round(acuracia[0] * 100, 10), '%. Epocas: ', epochs)

# salva o modelo
print('salvando o modelo como pkl')
#dump(model, open('fertility_tensor_model.pkl', 'wb'))

###
### --== random forest ==--
###

print('balanceando os dados para classificacao no random forest')
resampler = SMOTE()
csv.featuresbalanceadas, csv.classebalanceada = resampler.fit_resample(csv.features, csv.classe)
print('frequencia apos o balanceamento')
from collections import Counter
print(Counter(csv.classebalanceada))

# junta novamente, agora balanceados
datasetbalanceado = csv.featuresbalanceadas.join(csv.classebalanceada, how='left')
RFfeatures = datasetbalanceado.drop(columns=['Output'])
RFclasses = datasetbalanceado['Output']

print('separa dataset de treino e teste')
RFfeatures_train, RFfeatures_test, RFClass_train, RFClass_test = train_test_split(RFfeatures, RFclasses, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#Treinar o modelo de ML
rf.fit(RFfeatures_train, RFClass_train)

#Pretestar o modelo obtido
RFpredict = rf.predict(RFfeatures_test)
print('Classes previstas de acordo com as instancias separadas para testes:', RFpredict)

#Imprimir as classes reservadas para testes lado a lado com as classes previstas pelo modelo
i = 0
print('COMPARATIVO CLASSES TESTE x CLASSE PREVISTAS')
for i in range(0, len(RFClass_test)):
    print(RFClass_test.iloc[i], '-', RFpredict[i])

#Acurácia do modelo
from sklearn import metrics
print('Acurácia global:', metrics.accuracy_score(RFClass_test, RFpredict))

#Salvar o modelo
print('Salvando o modelo randomforest para o dataset fertility')
from pickle import dump
#dump(rf, open('fertility_random_forest_model.pkl','wb'))

print('fim do programa!')
