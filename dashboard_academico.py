import streamlit as st
import pandas as pd
import numpy as np

# CSS para customização de estilo
st.markdown("""
    <style>
        /* Estilo Geral do Corpo */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }

        /* Estilo do Cabeçalho */
        .header {
            background-color: #d71921;
            color: white;
            padding: 20px;
            text-align: center;
        }

        .header img {
            height: 50px;
            margin-bottom: 10px;
        }

        /* Estilo para Títulos */
        h1, h2 {
            text-align: center;
            color: #333;
            margin-top: 20px;
        }

        /* Estilo do Iframe */
        iframe {
            display: block;
            margin: 20px auto;
            border: 2px solid #d71921;
            border-radius: 8px;
            max-width: 90%;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Estilo para o Rodapé */
        .footer {
            background-color: #d71921;
            color: white;
            text-align: center;
            padding: 15px 0;
            font-size: 14px;
            position: fixed;
            width: 100%;
            bottom: 0;
        }

        /* Estilo para Parágrafos e Textos de Descrição */
        p {
            text-align: center;
            color: #555;
            font-size: 16px;
            margin: 10px 0;
        }
        
        /* Seção de Membros */
        .members {
            background-color: #ffffff;
            padding: 20px;
            margin: 20px;
            border: 1px solid #d71921;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Estrutura do App
st.markdown("""
<div class="header">
    <img src="https://univesp.br/sites/58f6506869226e9479d38201/theme/images/logo-univesp.png" alt="Logo UNIVESP">
    <h1>Dashboard Acadêmico UNIVESP</h1>
</div>
""", unsafe_allow_html=True)

# Título principal e descrição
st.title("Projeto Integrador IV")
st.write("Previsão de Temperaturas para Região de Sorocaba.")
st.write("Navegue e clique nos dados abaixo.")

# Dashboard Power BI embutido
st.markdown("""
    <iframe width="100%" height="800" src="https://app.powerbi.com/view?r=eyJrIjoiYWM2OGJmNjEtMDZlMi00Y2I5LWI2MjEtMTBmM2NiNzc2Y2Y0IiwidCI6IjM0NWRiODhhLTEzYzgtNDBmZS1iOGZiLTcxNDQxZDQ0NTEwOSJ9" frameborder="0" allowFullScreen="true"></iframe>
""", unsafe_allow_html=True)

# Seção de membros
st.markdown("""
<div class="members">
    <h5>Projeto Integrador em Computação IV da UNIVESP, dos cursos de graduação em Ciência de Dados e Engenharia da Computação, com análise de dados em escala do Instituto Nacional de Metereologia (INMET) de 2000 a 2025, desenvolvido por:</h5>
    <p>Alessandra Ghiraldi Ferraz</p>
    <p>Edgar Clemente Bispo</p>
    <p>Eduardo Fernandes Bueno</p>
    <p>Juergen Johann Moura Martins</p>
    <p>Marcus Vinicius Mathias</p>
    <p>Rafael Paraizo Bueno</p>
    <p>Roberto Akiba de Oliveira</p>
</div>
""", unsafe_allow_html=True)

# Rodapé
st.markdown("""
<div class="footer">
    &copy; 2024 UNIVESP - Universidade Virtual do Estado de São Paulo
</div>
""", unsafe_allow_html=True)