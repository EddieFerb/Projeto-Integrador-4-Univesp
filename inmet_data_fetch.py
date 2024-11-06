# import basedosdados as bd
# from google.cloud import bigquery

# # Billing ID deve ser uma string, então coloque entre aspas
# billing_id = '01F04B-45117D-223E0A'

# # Inicialize o cliente do BigQuery com a região correta
# client = bigquery.Client(location='us-central1')  # Altere para a região apropriada, se necessário

# query = """
#   SELECT
#     dados.ano as ano,
#     dados.data as data,
#     dados.hora as hora,
#     dados.id_estacao as id_estacao,
#     dados.precipitacao_total as precipitacao_total,
#     dados.pressao_atm_hora as pressao_atm_hora,
#     dados.pressao_atm_max as pressao_atm_max,
#     dados.pressao_atm_min as pressao_atm_min,
#     dados.radiacao_global as radiacao_global,
#     dados.temperatura_bulbo_hora as temperatura_bulbo_hora,
#     dados.temperatura_orvalho_hora as temperatura_orvalho_hora,
#     dados.temperatura_max as temperatura_max,
#     dados.temperatura_min as temperatura_min,
#     dados.temperatura_orvalho_max as temperatura_orvalho_max,
#     dados.temperatura_orvalho_min as temperatura_orvalho_min,
#     dados.umidade_rel_max as umidade_rel_max,
#     dados.umidade_rel_min as umidade_rel_min,
#     dados.umidade_rel_hora as umidade_rel_hora,
#     dados.vento_direcao as vento_direcao,
#     dados.vento_rajada_max as vento_rajada_max,
#     dados.vento_velocidade as vento_velocidade
#   FROM `basedosdados.br_inmet_bdmep.microdados` AS dados
# """

# # Correção: usar 'billing_project_id' como parâmetro
# df = bd.read_sql(query=query, billing_project_id=billing_id)

# # Exibir as primeiras linhas dos dados
# print(df.head())
import basedosdados as bd  # Certifique-se de que basedosdados está importado como bd
from google.cloud import bigquery

# ID de cobrança e ID do projeto
billing_id = '01F04B-45117D-223E0A'
project_id = 'forward-emitter-437218'

# Inicialize o cliente BigQuery com a região correta
client = bigquery.Client(project=project_id, location="southamerica-east1")

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
"""

# Execute a consulta com o billing_project_id especificado
df = bd.read_sql(query=query, billing_project_id=billing_id)

# Exibir as primeiras linhas para verificar a extração
print(df.head())