from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
import pickle

# Carrega os dados
with open(r'C:\UFJF\Eng.Software\IC\repos\mla-picker\src\data\data.pkl', 'rb') as f:  
  X_data_treinamento, y_data_treinamento, X_data_teste, y_data_teste = pickle.load(f)

# Divide os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_data_treinamento, y_data_treinamento, test_size=0.2, random_state=42)

# Cria o modelo de rede neural
#model = MLPClassifier(hidden_layer_sizes=(2, 2), max_iter=500)

model = MLPClassifier(hidden_layer_sizes=(2, 2), max_iter=500, warm_start=True, verbose= True, activation='tanh', solver='adam')
model.fit(X_train, y_train)
model._fit(X_train, y_train)
    
# Define o EarlyStopping:  interrompe o treinamento antes que o modelo comece a overfitting(sobreajuste) aos dados de treinamento
warnings.filterwarnings('ignore', category=ConvergenceWarning)
early_stopping = MLPClassifier(hidden_layer_sizes=(2, 2), max_iter=500, warm_start=True, verbose= True, activation='tanh', solver='adam')

early_stopping.fit(X_train, y_train)
# Treina o modelo com EarlyStopping
best_model = None
best_accuracy = 0
patience = 10
for i in range(500):
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pickle.loads(pickle.dumps(model))
        
    if i > patience and accuracy < best_accuracy:
        print(f"EarlyStopping: Iteration {i}, Best Accuracy: {best_accuracy}")
        break
        
    early_stopping.coefs_ = model.coefs_
    early_stopping.intercepts_ = model.intercepts_

    early_stopping.partial_fit(X_train, y_train, classes=np.unique(y_train))
    y_pred = early_stopping.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pickle.loads(pickle.dumps(early_stopping))
    
    if i > patience and accuracy < best_accuracy:
        print(f"EarlyStopping: Iteration {i}, Best Accuracy: {best_accuracy}")
        break
        
# Avalia o modelo com melhor acurácia
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Accuracy: {accuracy}")