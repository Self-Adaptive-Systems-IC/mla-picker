import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

##Matriz de confusao com base no atributo criticidade

# Carrega os dados do arquivo CSV
path = 'C:/UFJF/Eng.Software/IC/repos/mla-picker/src/data/dataset_train_test_10k_failures.csv'
data = pd.read_csv(path, decimal=",")

# Seleciona apenas as colunas relevantes
X = data[["criticality"]]
y = data["label"]

# Divide os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pré-processamento dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define o modelo da rede neural
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=[X_train.shape[1]]),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compila o modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define a técnica de early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=10)

# Treina o modelo
history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000, batch_size=32, callbacks=[early_stopping])

# Avalia o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Faz as previsões com o modelo treinado
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Exibe a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot da matriz de confusão
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()