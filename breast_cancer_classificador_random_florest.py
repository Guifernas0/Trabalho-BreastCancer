from sklearn.ensemble import RandomForestClassifier
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

#3 NORMALIZAR OS DADOS CATEGORICOS
dados_cat_rotulos = ['age','menopause','tumor-size','inv-nodes','node-caps','breast','breast-quad','irradiat']
dados_num = dados_atributos.drop(columns=dados_cat_rotulos)
dados_cat = dados_atributos[dados_cat_rotulos]
dados_cat_normalizados = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)
dados_atributos = dados_num.join(dados_cat_normalizados)

#4 BALANCEAR OS DADOS
resampler = SMOTE()
dados_atributos_balanceados, dados_classes_balanceados = resampler.fit_resample(dados_atributos, dados_classes)

#5 NORMALIZAR OS DADOS NUMERICOS
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_atributos_balanceados)
dump(modelo_normalizador, open('breast_cancer_normalizador.model','wb'))
dados_atributos_normalizados = modelo_normalizador.fit_transform(dados_atributos_balanceados)
dados_atributos_normalizados = pd.DataFrame(dados_atributos_normalizados, columns=dados_atributos.columns)

#6 HIPERPARAMETRIZACAO
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
#7 TREINAR DEFINITIVAMENTE E SALVAR O MODELO
rf = RandomForestClassifier(**rf_hiperparametros.best_params_)
cancer_rf = rf.fit(dados_atributos_normalizados, dados_classes_balanceados)
dump(cancer_rf, open('breast_cancer_rf.model', 'wb'))

#8 AVALIAR A ACURACIA DO MODELO
classes_preditas = cancer_rf.predict(dados_atributos_normalizados)
acuracia_cancer_rf = accuracy_score(dados_classes_balanceados, classes_preditas)
print('Acuracia (dados de treinamento):',acuracia_cancer_rf)
print('Acuracia (validacao cruzada):',rf_hiperparametros.best_score_)
