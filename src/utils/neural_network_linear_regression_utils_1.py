
##___________________________________________________________________________________________________________
## Realizar uma análise de regressão para prever o custo de reparo de uma máquina 
# com base nos atributos de tipo de falha, tempo de reparo e críticidade.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
# Carrega os dados do arquivo CSV
# path = 'C:/UFJF/Eng.Software/IC/repos/mla-picker/src/data/dataset_train_test_10k_failures.csv'
# data = pd.read_csv(path, decimal=",")

# # Seleciona apenas as colunas relevantes
# X = data[["type_of_failure", "time_repair", "criticality"]]
# y = data["cost"]

# # Divide os dados em treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.3, random_state=100)

# # Pré-processamento dos dados
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

def foo(X_train, y_train, X_test, y_test):
    # Define o modelo da rede neural
    model = keras.Sequential([
        layers.Dense(64, activation="tanh", input_shape=[X_train.shape[1]]),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    # Compila o modelo
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Define a técnica de early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)

    # Treina o modelo
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000, batch_size=32, callbacks=[early_stopping])

    # Avalia o modelo
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print(model.predict(X_test))
    # Faz as previsões com o modelo treinado
    y_pred = model.predict(X_test)
    # Exibe as métricas de regressão
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, recall_score
    from sklearn.metrics import precision_recall_fscore_support as score
    # precision, recall, fscore, support = score(y_test, y_pred)    
    # tp_i = []
    # for k in y_pred:
        # if(y_test.iloc[k] == 1):
            # tp_i.append(k)
    # print("R2 Score:", r2_score(y_test, tp_i))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
             
    # # Plota o gráfico de regressão
    plt.scatter(y_test, y_pred)
    plt.xlabel('Custo real')
    plt.ylabel('Custo previsto')
    plt.show()
    # print(y_test)
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy 


# # Define o modelo da rede neural
# model = keras.Sequential([
#     layers.Dense(64, activation="tanh", input_shape=[X_train.shape[1]]),
#     layers.Dense(32, activation="relu"),
#     layers.Dense(1)
# ])

# # Compila o modelo
# model.compile(optimizer="adam", loss="mean_squared_error")

# # Define a técnica de early stopping
# early_stopping = EarlyStopping(monitor="val_loss", patience=10)

# # Treina o modelo
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000, batch_size=32, callbacks=[early_stopping])

# # Avalia o modelo
# test_loss = model.evaluate(X_test, y_test)
# print("Test Loss:", test_loss)

# # Faz as previsões com o modelo treinado
# y_pred = model.predict(X_test)

# # Exibe as métricas de regressão
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# print("R2 Score:", r2_score(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("MSE:", mean_squared_error(y_test, y_pred))

# # Faz as previsões com o modelo treinado
# y_pred = model.predict(X_test)

# # Plota o gráfico de regressão
# plt.scatter(y_test, y_pred)
# plt.xlabel('Custo real')
# plt.ylabel('Custo previsto')
# plt.show()
