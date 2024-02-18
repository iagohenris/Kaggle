#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

teste = pd.read_csv(r'inputs\test.csv', sep=',')
treinamento = pd.read_csv(r'inputs\train.csv', sep=',')
treinamento.columns = treinamento.columns.str.upper()
teste.columns = teste.columns.str.upper()
#%%
print(treinamento.nunique())
print(treinamento.isnull().sum().sort_values(ascending=False))
print(treinamento.dtypes)
#%%
#juntando df de treino e teste
df_merged = pd.concat([treinamento, teste]).reset_index(drop=True)
#preenchendo idades faltantes com a mediana
mediana_idade = df_merged.AGE.median()
df_merged['AGE'] = df_merged.AGE.fillna(mediana_idade)
#preenchendo tarifas faltantes com a mediana
mediana_tarifa = df_merged.FARE.median()
df_merged['FARE'] = df_merged.FARE.fillna(mediana_tarifa)
#transformando sexo em 0 - masculino e 1 - feminino
df_merged['SEX'] = df_merged.SEX.map({'male':0,
                                      'female':1})
#separando teste e treinamento
treinamento = df_merged.iloc[:treinamento.shape[0]]
teste = df_merged.iloc[treinamento.shape[0]:]
#%%
#MODELO COM REGRESSÃO LOGISTICA
# criar um modelo de Regressão Logística
#separando variaveis explicativas do modelo e alvo
var_treinamento = treinamento[['PCLASS', 'SEX', 'FARE','AGE','SIBSP','PARCH']]
var_teste = teste[['PCLASS', 'SEX', 'FARE','AGE','SIBSP','PARCH']]
alvo = treinamento.SURVIVED
#fit modelo
modelo_log = LogisticRegression(solver='liblinear')
modelo_log.fit(var_treinamento, alvo)

# verificar a acurácia do modelo
acuracia_log = round(modelo_log.score(var_treinamento, alvo) * 100, 2)
print("Acurácia do modelo de Regressão Logística: {}".format(acuracia_log))
#rodando conjunto teste
modelo_teste = modelo_log.predict(var_teste)
resultado = pd.DataFrame({'PassengerID':teste.PASSENGERID,
                         'Survived':modelo_teste})
resultado.to_csv(r'outputs\result_lr.csv', index = False, sep=';')
#%%
# criar um modelo de árvore de decisão
modelo_arvore = DecisionTreeClassifier(max_depth=3)
modelo_arvore = modelo_arvore.fit(var_treinamento, alvo)

# verificar a acurácia do modelo
acuracia_arvore = round(modelo_arvore.score(var_treinamento, alvo) * 100, 2)
print("Acurácia do modelo de Árvore de Decisão: {}".format(acuracia_arvore))
arvore_teste = modelo_arvore.predict(var_teste)
resultado_arvore = pd.DataFrame({'PassengerID':teste.PASSENGERID,
                         'Survived':arvore_teste})
resultado_arvore.to_csv(r'outputs\result_tree.csv', index = False, sep=';')
#%%
resultado_train = modelo_log.predict(var_treinamento)
matriz_confusao = confusion_matrix(treinamento['SURVIVED'], resultado_train)
matriz_confusao
# %%
resultado_train = modelo_arvore.predict(var_treinamento)
matriz_confusao = confusion_matrix(treinamento['SURVIVED'], resultado_train)
matriz_confusao
#%%
# Exibir a matriz de confusão com melhores resultados usando Seaborn
sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
