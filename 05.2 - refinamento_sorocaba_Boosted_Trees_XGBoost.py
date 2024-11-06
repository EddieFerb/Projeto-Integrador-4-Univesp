from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Exemplo de dados: crie um DataFrame ou carregue seus dados aqui
# df = pd.read_csv('seus_dados.csv')  # Carregar seus dados
# X = df.drop(columns=['target'])  # Substitua 'target' pelo nome da sua coluna de destino
# y = df['target']

# Para fins de exemplo, vamos criar dados aleatórios
# X = np.random.rand(100, 10)  # 100 amostras e 10 características
# y = np.random.rand(100)       # 100 valores de destino

# Definir X e y
X = ...  # Defina X aqui
y = ...  # Defina y aqui

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo Boosted Trees
model_boosted = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Treinar o modelo
model_boosted.fit(X_train, y_train)

# Fazer previsões
y_pred_boosted = model_boosted.predict(X_test)

# Avaliar o modelo (imprimir RMSE - Root Mean Square Error)
rmse_boosted = np.sqrt(mean_squared_error(y_test, y_pred_boosted))
print(f"RMSE Boosted Trees: {rmse_boosted}")
