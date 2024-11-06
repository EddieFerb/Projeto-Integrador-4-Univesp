import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados do arquivo Excel
df = pd.read_excel('dados.xlsx')

# Remover espaços em branco nos nomes das colunas
df.columns = df.columns.str.strip()

# Remover linhas com NaN nas colunas relevantes
df = df.dropna(subset=['Dia', 'Umidade', 'Velocidade do Vento', 'Temperatura'])

# Garantir que a coluna 'Dia' seja do tipo numérico
df['Dia'] = pd.to_numeric(df['Dia'], errors='coerce')

# Remover quaisquer NaN resultantes da conversão
df = df.dropna(subset=['Dia', 'Umidade', 'Velocidade do Vento', 'Temperatura'])

# Usar 'Dia', 'Umidade', e 'Velocidade do Vento' como entradas
X = df[['Dia', 'Umidade', 'Velocidade do Vento']]
y = df['Temperatura']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo (imprime a pontuação R^2)
print("R^2 score:", model.score(X_test, y_test))

# Previsões para 2025
datas_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

# Criar um DataFrame para armazenar os dados de previsão para 2025
novos_dados = pd.DataFrame({
    'Dia': datas_2025.day,  # Extrair o dia do mês
    'Umidade': np.random.randint(40, 100, size=len(datas_2025)),  # Simulando umidade
    'Velocidade do Vento': np.random.randint(0, 20, size=len(datas_2025))  # Simulando velocidade do vento
})

# Fazer previsões para o ano de 2025
previsoes_2025 = model.predict(novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']])  # Usar todas as colunas relevantes

# Adicionar as previsões ao DataFrame
novos_dados['Temperatura Prevista'] = previsoes_2025

# Adicionar a coluna de data ao DataFrame
novos_dados['Data'] = datas_2025

# Exibir as previsões
print(novos_dados[['Data', 'Temperatura Prevista', 'Umidade', 'Velocidade do Vento']])

# Salvar previsões em um arquivo Excel
novos_dados.to_excel('previsoes_temperatura_2025.xlsx', index=False)

# Gerar gráfico das previsões
plt.figure(figsize=(12, 6))
plt.plot(novos_dados['Data'], novos_dados['Temperatura Prevista'], label='Temperatura Prevista', color='orange')
plt.title('Previsão de Temperatura para 2025')
plt.xlabel('Data')
plt.ylabel('Temperatura')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()  # Ajusta o layout para não cortar os rótulos
plt.show()  # Mostra o gráfico
