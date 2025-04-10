import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

# Carregar dados de 2022
df = pd.read_excel('dados_2022.xlsx')

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

# Gerar gráfico para 2022
datas_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# Simular umidade e velocidade do vento para 2022
novos_dados = pd.DataFrame({
    'Dia': datas_2022.day,
    'Mes': datas_2022.month,
    'Umidade': np.random.randint(40, 100, size=len(datas_2022)),
    'Velocidade do Vento': np.random.randint(0, 20, size=len(datas_2022))
})

# Escalar as entradas para 2022
novos_dados_scaled = scaler_X.transform(novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']])

# Fazer previsões com o modelo de rede neural
previsoes_nn_2022 = model_nn.predict(novos_dados_scaled)

# Normalizar as previsões para um intervalo de 0°C a 37°C
previsoes_nn_2022 = (previsoes_nn_2022 - previsoes_nn_2022.min()) / (previsoes_nn_2022.max() - previsoes_nn_2022.min()) * 37
previsoes_nn_2022 = previsoes_nn_2022.flatten()

# Adicionar previsões ao DataFrame e garantir que a temperatura esteja em ºC
novos_dados['Temperatura Prevista (ºC)'] = previsoes_nn_2022

# Caminho para salvar o arquivo
output_path_csv = './Inmet_Sorocaba/previsoes_temperatura_sorocaba_2022_nn.csv'
output_path_xlsx = './Inmet_Sorocaba/previsoes_temperatura_sorocaba_2022_nn.xlsx'

# Salvar previsões em CSV e Excel
novos_dados.to_csv(output_path_csv, index=False)
novos_dados.to_excel(output_path_xlsx, index=False)

# Mostrar temperatura máxima e mínima no terminal
temp_max = novos_dados['Temperatura Prevista (ºC)'].max()
temp_min = novos_dados['Temperatura Prevista (ºC)'].min()
print(f"Temperatura máxima prevista para 2022: {temp_max:.2f} ºC")
print(f"Temperatura mínima prevista para 2022: {temp_min:.2f} ºC")

# Verificação de salvamento
if os.path.exists(output_path_csv) and os.path.exists(output_path_xlsx):
    print(f"Previsões salvas com sucesso em:\n- CSV: {output_path_csv}\n- XLSX: {output_path_xlsx}")
else:
    print("Erro ao salvar os arquivos.")
    
# Visualização das previsões
plt.figure(figsize=(10, 6))
plt.plot(datas_2022, previsoes_nn_2022, label='Temperatura Prevista - Neural Network (ºC)', color='green')
plt.title('Previsão de Temperatura para Sorocaba em 2022 (Rede Neural)')
plt.xlabel('Data')
plt.ylabel('Temperatura (ºC)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
