from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Carrega os dados usando pandas
df = pd.read_csv(r'C:\UFJF\Eng.Software\IC\repos\mla-picker\src\data\dataset_train_test_10k_failures.csv', decimal=",")

# Divide os dados em features e target
X_data_treinamento = df.drop(columns=['cost']).values
y_data_treinamento = df['cost'].values

# técnica de pré-processamento de dados que garante que os dados tenham a mesma escala e distribuição
scaler = StandardScaler()
X_data_treinamento = scaler.fit_transform(X_data_treinamento)

# Divide os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_data_treinamento, y_data_treinamento, test_size=0.3, random_state=42)

# Cria o modelo de rede neural
model = MLPRegressor()
params = params = {'hidden_layer_sizes': [(10,), (50,), (100,), (150,)],
          'activation': ['relu', 'tanh'],
          'solver': ['adam', 'sgd'],
          'learning_rate': ['constant', 'invscaling', 'adaptive'],
          'max_iter': [100, 200, 500, 1000],
          'alpha': [0.0001, 0.001, 0.01],
          'batch_size': [32, 64, 128],
          'verbose': [True]}
grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')

# treina o modelo usando o grid search
grid_search.fit(X_train, y_train)

# Avalia o modelo com melhor acurácia
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
print(f"Best Parameters: {grid_search.best_params_}")

