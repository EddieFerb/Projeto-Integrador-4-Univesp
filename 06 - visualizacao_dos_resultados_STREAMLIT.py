import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Inserindo o logotipo da UNIVESP no centro da página
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("logo-univesp_completo_cor-negativo.png", width=200)
st.markdown("</div>", unsafe_allow_html=True)

# Título do App, centralizado
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Dashboard Acadêmico UNIVESP</h1>
        <h2>Projeto Integrador IV</h2>
        <h3>Previsão de Temperaturas para a Região de Sorocaba</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Lista de arquivos
files = [
    "01- previsoes.py",
    "02 - previsoes_2025.py",
    "03 - previsoes_2025_grafico.py",
    "04 - previsoes_2025_last.py",
    "05 - refinamento_sorocaba.py",
    "05.5 - refinamento_sorocaba_dia_a_dia_2025.py",
    "05.6 - refinamento_sorocaba_dia_a_dia_2020.py",
    "05.6 - refinamento_sorocaba_dia_a_dia_2021.py",
    "05.6 - refinamento_sorocaba_dia_a_dia_2022.py",
    "05.6 - refinamento_sorocaba_dia_a_dia_2023.py",
    "05.7 - refinamento_nacional_dia_a_dia_2025_bigquery.png"
]

# Criando uma seleção de arquivos
selected_file = st.selectbox("Selecione um arquivo para exibir o gráfico:", files)

# Gráfico interativo de temperatura para 365 dias
temperatura_data = pd.DataFrame({
    '365 dias': range(1, 366),
    'temperatura (°C)': np.random.uniform(0, 37, 365)
})

st.line_chart(temperatura_data.set_index('365 dias'))

# Verifica se o arquivo é uma imagem e carrega
if selected_file.endswith('.png'):
    if os.path.exists(selected_file):
        image = Image.open(selected_file)
        st.image(image, caption=f"Visualização de {selected_file}")
    else:
        st.error(f"Imagem {selected_file} não encontrada.")
else:
    # Exibe o conteúdo do arquivo .py selecionado
    with open(selected_file, 'r') as file:
        code = file.read()
    st.code(code, language='python')

# Nomes dos membros do projeto (no final da página)
st.markdown("""
**Desenvolvido por:**
- Alessandra Ghiraldi Ferraz
- Edgar Clemente Bispo
- Eduardo Fernandes Bueno
- Juergen Johann Moura Martins
- Marcus Vinicius Mathias
- Rafael Paraizo Bueno
- Roberto Akiba de Oliveira
""")