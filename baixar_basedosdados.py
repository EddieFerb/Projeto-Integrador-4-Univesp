from google.cloud import bigquery
import basedosdados as bd

# Configurações de BigQuery com project_id
project_id = 'XXXXXXXXXXXXXXXXX'  # Use o project_id
billing_id = 'XXXXXXXXXXXXXXXXX'  # billing_project_id deve ser project_id

# Inicialize o cliente BigQuery
client = bigquery.Client(project=project_id)

# Query para extrair dados
query = """
  SELECT
    dados.ano as ano,
    dados.data as data,
    dados.hora as hora,
    dados.id_estacao as id_estacao,
    dados.precipitacao_total as precipitacao_total,
    dados.pressao_atm_hora as pressao_atm_hora,
    dados.pressao_atm_max as pressao_atm_max,
    dados.pressao_atm_min as pressao_atm_min,
    dados.radiacao_global as radiacao_global,
    dados.temperatura_bulbo_hora as temperatura_bulbo_hora,
    dados.temperatura_orvalho_hora as temperatura_orvalho_hora,
    dados.temperatura_max as temperatura_max,
    dados.temperatura_min as temperatura_min,
    dados.temperatura_orvalho_max as temperatura_orvalho_max,
    dados.temperatura_orvalho_min as temperatura_orvalho_min,
    dados.umidade_rel_max as umidade_rel_max,
    dados.umidade_rel_min as umidade_rel_min,
    dados.umidade_rel_hora as umidade_rel_hora,
    dados.vento_direcao as vento_direcao,
    dados.vento_rajada_max as vento_rajada_max,
    dados.vento_velocidade as vento_velocidade
  FROM `basedosdados.br_inmet_bdmep.microdados` AS dados
  WHERE dados.ano >= 2000 AND dados.ano <= 2020
"""

# Execute a consulta
df = bd.read_sql(query=query, billing_project_id=billing_id)

# Exibir as primeiras linhas para verificar a extração
print(df.head())

# Salvar o DataFrame em um arquivo CSV
df.to_csv('/Users/eddieferb/Inmet_Sorocaba/dados_inmet_2000_2020.csv', index=False)
