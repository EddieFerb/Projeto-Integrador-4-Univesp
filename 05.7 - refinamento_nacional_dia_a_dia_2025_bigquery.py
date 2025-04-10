import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

# Carregar dados de 2000 a 2020
df = pd.read_csv('dados_inmet_2000_2020.csv')

# Limpeza e preparação dos dados
df.columns = df.columns.str.strip()
df = df.dropna(subset=['data', 'temperatura_max', 'umidade_rel_hora', 'vento_velocidade'])
df['Dia'] = pd.to_datetime(df['data']).dt.day
df['Mes'] = pd.to_datetime(df['data']).dt.month

# Entradas e Saída
X = df[['Dia', 'umidade_rel_hora', 'vento_velocidade']]
y = df['temperatura_max']

# Dividir dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar apenas as entradas
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Modelo de Rede Neural usando Keras
model_nn = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(units=64, activation='relu'),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

model_nn.compile(optimizer='adam', loss='mean_squared_error')
model_nn.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test))

# Gerar dados simulados para 2025
datas_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
novos_dados = pd.DataFrame({
    'Dia': datas_2025.day,
    'Mes': datas_2025.month,
    'umidade_rel_hora': np.random.randint(40, 100, size=len(datas_2025)),
    'vento_velocidade': np.random.randint(0, 20, size=len(datas_2025))
})

# Escalar as entradas para 2025
novos_dados_scaled = scaler_X.transform(novos_dados[['Dia', 'umidade_rel_hora', 'vento_velocidade']])

# Fazer previsões com o modelo de rede neural
previsoes_nn_2025 = model_nn.predict(novos_dados_scaled)

# Normalizar as previsões para um intervalo de 0°C a 37°C
previsoes_nn_2025 = (previsoes_nn_2025 - previsoes_nn_2025.min()) / (previsoes_nn_2025.max() - previsoes_nn_2025.min()) * 37
previsoes_nn_2025 = previsoes_nn_2025.flatten()

# Adicionar previsões ao DataFrame e garantir que a temperatura esteja em ºC
novos_dados['Temperatura Prevista (ºC)'] = previsoes_nn_2025
novos_dados['Data'] = datas_2025

# Caminho para salvar os arquivos
output_path_csv = './Inmet_Sorocaba/05.7 - refinamento_nacional_dia_a_dia_2025_bigquery.csv'
output_path_xlsx = './Inmet_Sorocaba/05.7 - refinamento_nacional_dia_a_dia_2025_bigquery.xlsx'

# Salvar previsões em CSV e Excel
novos_dados.to_csv(output_path_csv, index=False)
novos_dados.to_excel(output_path_xlsx, index=False)

# Mostrar temperatura máxima e mínima no terminal
temp_max = novos_dados['Temperatura Prevista (ºC)'].max()
temp_min = novos_dados['Temperatura Prevista (ºC)'].min()
print(f"Temperatura máxima prevista para 2025: {temp_max:.2f} ºC")
print(f"Temperatura mínima prevista para 2025: {temp_min:.2f} ºC")

# Verificação de salvamento
if os.path.exists(output_path_csv) and os.path.exists(output_path_xlsx):
    print(f"Previsões salvas com sucesso em:\n- CSV: {output_path_csv}\n- XLSX: {output_path_xlsx}")
else:
    print("Erro ao salvar os arquivos.")
    
# Visualização das previsões
plt.figure(figsize=(10, 6))
plt.plot(datas_2025, previsoes_nn_2025, label='Temperatura Prevista - Neural Network (ºC)', color='blue')
plt.title('Previsão de Temperatura para Brasil em 2025 (Rede Neural)')
plt.xlabel('Data')
plt.ylabel('Temperatura (ºC)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()