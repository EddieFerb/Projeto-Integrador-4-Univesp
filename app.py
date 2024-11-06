import streamlit as st
from streamlit.components.v1 import html

# Define uma função para carregar o conteúdo do HTML com o CSS
def render_html_page():
    with open("index.html", "r") as html_file:
        html_content = html_file.read()
    
    # Carregar o estilo CSS
    st.markdown("<style>" + open("styles.css").read() + "</style>", unsafe_allow_html=True)
    
    # Renderizar o conteúdo HTML na página
    html(html_content, height=800)

# Define uma função para o script Streamlit interativo
def render_streamlit_page():
    # Importa o conteúdo do arquivo Python com o código interativo
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", "06 - visualizacao_dos_resultados_STREAMLIT.py")
    streamlit_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streamlit_app)

# Cria as páginas no Streamlit
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma página:", ("Página HTML", "Página Interativa"))

if page == "Página HTML":
    render_html_page()
elif page == "Página Interativa":
    render_streamlit_page()