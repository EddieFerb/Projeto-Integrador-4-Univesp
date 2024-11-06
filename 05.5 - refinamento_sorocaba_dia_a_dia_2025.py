# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from sklearn.ensemble import RandomForestRegressor

# # Carregar dados de 2020 a 2024
# df = pd.read_excel('dados.xlsx')

# # Limpeza e preparação dos dados
# df.columns = df.columns.str.strip()
# df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])
# df['Dia'] = pd.to_numeric(df['Dia'], errors='coerce')
# df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])

# # Entradas e Saída
# X = df[['Dia', 'Umidade', 'Velocidade do Vento']]
# y = df['Temperatura']

# # Dividir dados em conjuntos de treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Escalar os dados
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Modelo de Rede Neural usando Keras
# model_nn = Sequential()
# model_nn.add(Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
# model_nn.add(Dropout(0.2))
# model_nn.add(Dense(units=64, activation='relu'))
# model_nn.add(Dropout(0.2))
# model_nn.add(Dense(units=32, activation='relu'))
# model_nn.add(Dense(units=1))  # Saída para prever a temperatura

# model_nn.compile(optimizer='adam', loss='mean_squared_error')
# model_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test))

# # Fazer previsões para 2025
# datas_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

# # Simular umidade e velocidade do vento para 2025
# novos_dados = pd.DataFrame({
#     'Dia': datas_2025.day,
#     'Mes': datas_2025.month,
#     'Umidade': np.random.randint(40, 100, size=len(datas_2025)),
#     'Velocidade do Vento': np.random.randint(0, 20, size=len(datas_2025))
# })

# novos_dados_scaled = scaler.transform(novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']])
# previsoes_nn_2025 = model_nn.predict(novos_dados_scaled)

# # Adicionar previsões ao DataFrame e garantir que a temperatura esteja em ºC
# novos_dados['Temperatura Prevista (ºC)'] = previsoes_nn_2025

# # Salvar previsões em um arquivo Excel
# novos_dados.to_excel('/Users/eddieferb/Inmet_Sorocaba/previsoes_temperatura_sorocaba_2025_nn.xlsx', index=False)

# # Visualização das previsões
# plt.figure(figsize=(10, 6))
# plt.plot(datas_2025, previsoes_nn_2025, label='Temperatura Prevista - Neural Network (ºC)', color='blue')
# plt.title('Previsão de Temperatura para Sorocaba em 2025 (Rede Neural)')
# plt.xlabel('Data')
# plt.ylabel('Temperatura (ºC)')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.legend()
# plt.show()
# import numpy as np
# import pandas as pd

# # Supondo que o modelo já tenha sido treinado e os dados estejam preparados
# # Definindo os dados de entrada para a previsão
# dados = novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']]

# # Aqui é feita a previsão da temperatura
# temp_prevista = model_nn.predict(dados)

# # Reescalando a previsão para o intervalo correto (0°C a 37°C)
# # Ajuste de normalização, se necessário
# temp_prevista = temp_prevista * (37 - 0) + 0

# # Limitando a temperatura prevista para ficar dentro do intervalo de 0°C a 37°C
# temp_prevista = np.clip(temp_prevista, 0, 37)

# # Criando um DataFrame para salvar as previsões em uma planilha
# df = pd.DataFrame(temp_prevista, columns=['Temperatura Prevista (ºC)'])

# # Salvando o DataFrame como um arquivo Excel na pasta especificada
# df.to_excel('/Users/eddieferb/Inmet_Sorocaba/temperatura_prevista.xlsx', index=False)

# print("Previsões de temperatura salvas com sucesso em '/Users/eddieferb/Inmet_Sorocaba/temperatura_prevista.xlsx'")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

# Carregar dados de 2020 a 2024
df = pd.read_excel('dados.xlsx')

# Limpeza e preparação dos dados
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])
df['Dia'] = pd.to_numeric(df['Dia'], errors='coerce')
df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])

# Entradas e Saída
X = df[['Dia', 'Umidade', 'Velocidade do Vento']]
y = df['Temperatura']

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

# Gerar previsões para 2025
datas_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

# Simular umidade e velocidade do vento para 2025
novos_dados = pd.DataFrame({
    'Dia': datas_2025.day,
    'Mes': datas_2025.month,
    'Umidade': np.random.randint(40, 100, size=len(datas_2025)),
    'Velocidade do Vento': np.random.randint(0, 20, size=len(datas_2025))
})

# Escalar as entradas para 2025
novos_dados_scaled = scaler_X.transform(novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']])

# Fazer previsões com o modelo de rede neural
previsoes_nn_2025 = model_nn.predict(novos_dados_scaled)

# Normalizar as previsões para um intervalo de 0°C a 37°C
previsoes_nn_2025 = (previsoes_nn_2025 - previsoes_nn_2025.min()) / (previsoes_nn_2025.max() - previsoes_nn_2025.min()) * 37
previsoes_nn_2025 = previsoes_nn_2025.flatten()

# Adicionar previsões ao DataFrame e garantir que a temperatura esteja em ºC
novos_dados['Temperatura Prevista (ºC)'] = previsoes_nn_2025

# Salvar previsões em um arquivo Excel
novos_dados.to_excel('/Users/eddieferb/Inmet_Sorocaba/previsoes_temperatura_sorocaba_2025_nn.xlsx', index=False)

# Adicionar previsões ao DataFrame e garantir que a temperatura esteja em ºC
novos_dados['Temperatura Prevista (ºC)'] = previsoes_nn_2025

# Caminho para salvar o arquivo
output_path_csv = '/Users/eddieferb/Inmet_Sorocaba/previsoes_temperatura_sorocaba_2025_nn.csv'
output_path_xlsx = '/Users/eddieferb/Inmet_Sorocaba/previsoes_temperatura_sorocaba_2025_nn.xlsx'

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
plt.title('Previsão de Temperatura para Sorocaba em 2025 (Rede Neural)')
plt.xlabel('Data')
plt.ylabel('Temperatura (ºC)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
