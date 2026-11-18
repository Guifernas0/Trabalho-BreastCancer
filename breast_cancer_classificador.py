from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from pickle import dump

#1 CARREGAR OS DADOS
dados = pd.read_csv('breast-cancer.csv', sep=',')

#2 SEPARAR ATRIBUTOS E CLASSE
dados_atributos = dados.drop(columns = ['Class'])
dados_classes = dados['Class']

#3 NORMALIZAR OS DADOS CATEGÓRICOS
dados_cat_rotulos = ['age','menopause','tumor-size','inv-nodes','node-caps','breast','breast-quad','irradiat']
dados_num = dados_atributos.drop(columns=dados_cat_rotulos)
dados_cat = dados_atributos[dados_cat_rotulos]
dados_cat_normalizados = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)
dados_atributos = dados_num.join(dados_cat_normalizados)

#4 BALANCEAR OS DADOS
resampler = SMOTE()
dados_atributos_balanceados, dados_classes_balanceados = resampler.fit_resample(dados_atributos, dados_classes)

#5 NORMALIZAR OS DADOS NUMÉRICOS
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_atributos_balanceados)
dump(modelo_normalizador, open('breast_cancer_normalizador.model','wb'))
dados_atributos_normalizados = modelo_normalizador.fit_transform(dados_atributos_balanceados)
dados_atributos_normalizados = pd.DataFrame(dados_atributos_normalizados, columns=dados_atributos.columns)

#6 HIPERPARAMETRIZAÇÃO DE CADA ALGORITMO

#6.1 RANDOM FOREST
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 300, num = 10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10,num=5)]
max_depth = [int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]
max_features = ['sqrt', 'log2', None]
rf_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}
rf = RandomForestClassifier()
rf_hiperparametros=RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)
rf_hiperparametros.fit(dados_atributos_normalizados,dados_classes_balanceados)
rf = RandomForestClassifier(**rf_hiperparametros.best_params_)
cancer_rf = rf.fit(dados_atributos_normalizados, dados_classes_balanceados)
print('Random Forest - Acuracia:', rf_hiperparametros.best_score_)

#6.2 DECISION TREE
criterion = ['gini', 'entropy','log_loss']
splitter = ['best', 'random']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10,num=5)]
tree_grid = {
    'criterion': criterion,
    'splitter': splitter,
    'min_samples_split': min_samples_split
}
tree = DecisionTreeClassifier()
tree_hiperparametros=RandomizedSearchCV(
    estimator=tree,
    param_distributions=tree_grid,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)
tree_hiperparametros.fit(dados_atributos_normalizados,dados_classes_balanceados)
tree = DecisionTreeClassifier(**tree_hiperparametros.best_params_)
cancer_tree = tree.fit(dados_atributos_normalizados, dados_classes_balanceados)
print('Decision Tree - Acuracia:', tree_hiperparametros.best_score_)

#6.3 KNN
n_neighbors = [int(x) for x in np.linspace(start=1, stop=30, num=10)]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
knn_grid = {
    'n_neighbors': n_neighbors,
    'weights': weights,
    'metric': metric
}
knn = KNeighborsClassifier()
knn_hiperparametros=RandomizedSearchCV(
    estimator=knn,
    param_distributions=knn_grid,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)
knn_hiperparametros.fit(dados_atributos_normalizados,dados_classes_balanceados)
knn = KNeighborsClassifier(**knn_hiperparametros.best_params_)
cancer_knn = knn.fit(dados_atributos_normalizados, dados_classes_balanceados)
print('KNN - Acuracia:', knn_hiperparametros.best_score_)

#6.4 SVM
C = [0.01, 0.1, 1, 10, 100]
kernel = ['linear', 'rbf', 'poly', 'sigmoid']
gamma = ['scale', 'auto']
svm_grid = {
    'C': C,
    'kernel': kernel,
    'gamma': gamma
}
svm = SVC()
svm_hiperparametros=RandomizedSearchCV(
    estimator=svm,
    param_distributions=svm_grid,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)
svm_hiperparametros.fit(dados_atributos_normalizados,dados_classes_balanceados)
svm = SVC(**svm_hiperparametros.best_params_)
cancer_svm = svm.fit(dados_atributos_normalizados, dados_classes_balanceados)
print('SVM - Acuracia:', svm_hiperparametros.best_score_)

#7 ESCOLHER O MODELO DE MELHOR ACURÁCIA
#A comparação usa a acurácia da validação cruzada (best_score_) do RandomizedSearchCV,
#pois ela é medida em dados que o modelo não viu durante cada rodada de treino
modelos = {
    'Random Forest': (rf_hiperparametros.best_score_, cancer_rf),
    'Decision Tree': (tree_hiperparametros.best_score_, cancer_tree),
    'KNN': (knn_hiperparametros.best_score_, cancer_knn),
    'SVM': (svm_hiperparametros.best_score_, cancer_svm)
}
melhor_nome = max(modelos, key=lambda nome: modelos[nome][0])
melhor_acuracia, cancer_modelo = modelos[melhor_nome]
print('Melhor modelo:', melhor_nome, '- Acuracia:', melhor_acuracia)

#Salvar o melhor modelo para uso posterior
dump(cancer_modelo, open('breast_cancer_modelo.model', 'wb'))

#8 AVALIAR A ACURÁCIA DO MELHOR MODELO
classes_preditas = cancer_modelo.predict(dados_atributos_normalizados)
acuracia_cancer_modelo = accuracy_score(dados_classes_balanceados, classes_preditas)
print('Acuracia nos dados de treinamento:',acuracia_cancer_modelo)
