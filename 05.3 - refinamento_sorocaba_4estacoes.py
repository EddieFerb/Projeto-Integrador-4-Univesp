import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Carregar os dados do arquivo Excel com dados de 2020 a 2024
df = pd.read_excel('dados.xlsx')

# Remover espaços em branco nos nomes das colunas
df.columns = df.columns.str.strip()

# Remover linhas com NaN nas colunas relevantes
df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])

# Garantir que a coluna 'Dia' seja do tipo numérico
df['Dia'] = pd.to_numeric(df['Dia'], errors='coerce')

# Remover quaisquer NaN resultantes da conversão
df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])

# Ler colunas 'Dia', 'Umidade', 'Velocidade do Vento' e 'Temperatura'
X = df[['Dia', 'Umidade', 'Velocidade do Vento']]  # Entradas
y = df['Temperatura']  # Saída

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Criar o modelo de Boosted Tree Regression
boosted_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
boosted_model.fit(X_train, y_train)

# Fazer previsões com Boosted Tree Regression
y_pred_boosted = boosted_model.predict(X_test)

# Avaliar o modelo Boosted Tree (imprime o erro quadrático médio)
print("Boosted Tree Regression - MSE:", mean_squared_error(y_test, y_pred_boosted))

# 2. Criar o modelo de Neural Network Regression (MLP Regressor)
nn_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Fazer previsões com Neural Network Regression
y_pred_nn = nn_model.predict(X_test)

# Avaliar o modelo Neural Network (imprime o erro quadrático médio)
print("Neural Network Regression - MSE:", mean_squared_error(y_test, y_pred_nn))

# Previsões para 2025 segmentado por estações do ano
datas_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

# Função para identificar a estação do ano com base no mês
def estacao_do_ano(mes):
    if mes in [12, 1, 2]:
        return 'Verão'
    elif mes in [3, 4, 5]:
        return 'Outono'
    elif mes in [6, 7, 8]:
        return 'Inverno'
    elif mes in [9, 10, 11]:
        return 'Primavera'

# Criar um DataFrame para armazenar os dados de previsão para 2025
novos_dados = pd.DataFrame({
    'Dia': datas_2025.day,  # Extrair o dia do mês
    'Mes': datas_2025.month,
    'Umidade': np.random.randint(40, 100, size=len(datas_2025)),  # Simulando umidade
    'Velocidade do Vento': np.random.randint(0, 20, size=len(datas_2025))  # Simulando velocidade do vento
})

# Adicionar a estação do ano ao DataFrame
novos_dados['Estacao'] = novos_dados['Mes'].apply(estacao_do_ano)

# Fazer previsões para 2025 usando o modelo de Boosted Tree Regression
previsoes_boosted = boosted_model.predict(novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']])

# Fazer previsões para 2025 usando o modelo de Neural Network Regression
previsoes_nn = nn_model.predict(novos_dados[['Dia', 'Umidade', 'Velocidade do Vento']])

# Adicionar as previsões ao DataFrame
novos_dados['Temperatura Prevista - Boosted'] = previsoes_boosted
novos_dados['Temperatura Prevista - NN'] = previsoes_nn

# Adicionar a coluna de data ao DataFrame
novos_dados['Data'] = datas_2025

# Exibir as previsões
print(novos_dados[['Data', 'Estacao', 'Temperatura Prevista - Boosted', 'Temperatura Prevista - NN', 'Umidade', 'Velocidade do Vento']])

# Salvar previsões em um arquivo Excel
novos_dados.to_excel('previsoes_temperatura_2025_refinado_sazonal.xlsx', index=False)

# Gerar gráficos das previsões para 2025 segmentado por estação

plt.figure(figsize=(14, 10))

# Gráfico de previsões com Boosted Tree segmentado por estação
plt.subplot(2, 1, 1)
for estacao in ['Verão', 'Outono', 'Inverno', 'Primavera']:
    plt.plot(novos_dados[novos_dados['Estacao'] == estacao]['Data'], 
             novos_dados[novos_dados['Estacao'] == estacao]['Temperatura Prevista - Boosted'], 
             label=f'Temperatura Prevista - Boosted ({estacao})')
plt.title('Previsão de Temperatura para 2025 - Boosted Tree Regression (Sazonal)')
plt.xlabel('Data')
plt.ylabel('Temperatura')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

# Gráfico de previsões com Neural Network segmentado por estação
plt.subplot(2, 1, 2)
for estacao in ['Verão', 'Outono', 'Inverno', 'Primavera']:
    plt.plot(novos_dados[novos_dados['Estacao'] == estacao]['Data'], 
             novos_dados[novos_dados['Estacao'] == estacao]['Temperatura Prevista - NN'], 
             label=f'Temperatura Prevista - NN ({estacao})')
plt.title('Previsão de Temperatura para 2025 - Neural Network Regression (Sazonal)')
plt.xlabel('Data')
plt.ylabel('Temperatura')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

plt.tight_layout()  # Ajusta o layout para não cortar os rótulos
plt.show()  # Mostra o gráfico