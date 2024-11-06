import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar os dados do arquivo Excel (substitua pelo seu arquivo de dados de 2020-2024)
df = pd.read_excel('dados.xlsx')

# Limpeza dos dados
df.columns = df.columns.str.strip()  # Remover espaços nas colunas
df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])

# Garantir que a coluna 'Dia' seja numérica
df['Dia'] = pd.to_numeric(df['Dia'], errors='coerce')
df = df.dropna(subset=['Dia', 'Temperatura', 'Umidade', 'Velocidade do Vento'])

# Definir entradas e saídas
X = df[['Dia', 'Umidade', 'Velocidade do Vento']]
y = df['Temperatura']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
