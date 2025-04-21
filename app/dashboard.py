import os
import sys
import time
import re
import streamlit as st
from PIL import Image

image = Image.open("data/header.png")

# Configurar a página com título e ícone
st.set_page_config(
    page_title="Jala Scholar",
    page_icon="data/logo.png",
    layout="wide"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mostra tela de carregamento antes de importar módulos pesados
with st.spinner("Carregando componentes do sistema..."):
    from src.search.semantic_search import search, load_embeddings
    from src.visualization.cluster_viz import plot_clean_embeddings, plot_grouped_embeddings

# Caminhos dos arquivos
PROCESSED_DOCS_FOLDER = "data/processed/"


def display_document_content(doc_name):
    """Exibe o conteúdo de um documento de forma formatada"""
    doc_path = os.path.join(PROCESSED_DOCS_FOLDER, doc_name)

    if not os.path.exists(doc_path):
        st.error(f"❌ O arquivo {doc_name} não foi encontrado.")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Criando um container para o conteúdo do documento
    doc_container = st.container()
    
    with doc_container:
        # Cabeçalho com o nome do documento formatado
        clean_doc_name = doc_name.replace('.txt', '').replace('_', ' ').title()
        st.markdown(f"### 📄 {clean_doc_name}")
        
        # Opções de tema para visualização de documentos
        themes = {
            "Escuro": {"bg": "#1E1E1E", "text": "#FFFFFF", "highlight": "#FFD700", "border": "#444444"},
            "Claro": {"bg": "#FFFFFF", "text": "#333333", "highlight": "#4B0082", "border": "#DDDDDD"},
            "Terminal": {"bg": "#000000", "text": "#00FF00", "highlight": "#FF5733", "border": "#333333"},
            "Azul": {"bg": "#0A1929", "text": "#E6F1FF", "highlight": "#5CBBF6", "border": "#2D4B6D"}
        }
        
        # Seletor de tema na barra lateral
        if 'doc_theme' not in st.session_state:
            st.session_state.doc_theme = "Escuro"
            
        with st.sidebar:
            st.session_state.doc_theme = st.selectbox("Tema do documento:", 
                                                     list(themes.keys()),
                                                     index=list(themes.keys()).index(st.session_state.doc_theme),
                                                     key="theme_selector")
        
        # Obter cores do tema atual
        theme = themes[st.session_state.doc_theme]
                
        # Adiciona um card com borda e padding para o texto
        st.markdown(f"""
        <style>
        .document-card {{
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            padding: 20px;
            background-color: {theme["bg"]};
            color: {theme["text"]};
            max-height: 600px;
            overflow-y: auto;
            font-family: 'Source Code Pro', 'Courier New', monospace;
            white-space: pre-wrap;
            line-height: 1.6;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .document-card h4 {{
            color: {theme["text"]};
            border-bottom: 1px solid {theme["border"]};
            padding-bottom: 8px;
            margin-bottom: 15px;
        }}
        .highlight-term {{
            background-color: {theme["highlight"]};
            color: {theme["bg"]};
            padding: 0 3px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .document-card p {{
            margin-bottom: 12px;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        highlight = st.checkbox("Destacar termos da busca", value=st.session_state.highlight, key="highlight_terms_doc")
        st.session_state.highlight = highlight
        
        # Destacar termos da busca se solicitado
        if st.session_state.query and st.session_state.highlight:
            highlighted_content = content
            for term in st.session_state.query.split():
                if len(term) > 2:
                    highlighted_content = highlighted_content.replace(
                        term, 
                        f'<span style="background-color: {theme["highlight"]};">{term}</span>'
                    )
                    st.markdown(f'<div class="document-card">{highlighted_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="document-card">{content}</div>', unsafe_allow_html=True)


def run_dashboard():
    """Inicia o dashboard do Streamlit"""

    # Inicializar variáveis de estado da sessão
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'doc_options' not in st.session_state:
        st.session_state.doc_options = {}
    if 'highlight' not in st.session_state:
        st.session_state.highlight = False
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Jala Header", use_container_width=True)
    
    st.title("📚 Dashboard de Busca de Documentação Técnica")

    # Carregar embeddings com indicador de progresso
    with st.spinner("🧠 Carregando embeddings... Isso pode levar alguns segundos."):
        doc_names, doc_embeddings = load_embeddings()
    

    if doc_names is None or doc_embeddings is None:
        st.error("⚠ Erro ao carregar embeddings. Execute `generate_embeddings.py` primeiro.")
        return
    else:
        st.success(f"✅ {len(doc_names)} documentos carregados com sucesso!")

    tab1, tab2 = st.tabs(["🔍 Busca Semântica", "📊 Visualização dos Clusters"])

    with tab1:
        # Busca Semântica        

        # Função para atualizar a busca
        def update_search():
            st.session_state.results = search(st.session_state.query, doc_names, doc_embeddings)
            if st.session_state.results:
                st.session_state.doc_options = {
                    f"{doc} (Similaridade: {score:.4f})": doc 
                    for doc, score in st.session_state.results
                }

        # Campo de busca com valor persistente
        query = st.text_input(
            "Digite sua busca:", 
            value=st.session_state.query,
            key="query_input"
        )

        # Atualizar a variável de estado
        st.session_state.query = query
        
        if st.button("Buscar", key="search_btn"):
            if st.session_state.query:
                update_search()
            else:
                st.warning("⚠ Por favor, digite uma consulta válida.")
        
        # Mostrar resultados se existirem
        if st.session_state.results:
            st.write("**📌 Resultados mais relevantes:**")
            
            # Exibir os documentos em um menu dropdown
            selected_doc = st.selectbox(
                "Selecione um documento para visualizar:", 
                list(st.session_state.doc_options.keys()),
                key="doc_selector"
            )
            
            # Exibir o conteúdo do documento selecionado
            if selected_doc:
                st.session_state.current_doc = st.session_state.doc_options[selected_doc]
                display_document_content(st.session_state.current_doc)
        elif st.session_state.query and st.session_state.results == []:
            st.warning("⚠ Nenhum documento relevante encontrado.")

        # Limpa o conteúdo do documento (basicamente ctrl + l do terminal)
        if st.session_state.current_doc:
            if st.button("Limpar Conteúdo", key="clear_page_btn"):
                st.session_state.current_doc = None
                st.session_state.query = ""
                st.session_state.results = None
                st.session_state.doc_options = {}
                st.session_state.highlight = False
                st.rerun()
    with tab2:
        # Visualização dos Clusters
        st.subheader("📊 Visualização dos Clusters")
        
        # Adiciona opção de escolher entre visualização simples e visualização por clusters
        visualization_type = st.radio("Escolha o tipo de visualização:", 
                                    ["Visualização Simples", "Visualização por Clusters"],
                                    horizontal=True,
                                    key="viz_type")
        
        st.write("Escolha o método de redução de dimensionalidade:")
        reduction_method = st.selectbox("", ["- Selecione -", "tsne", "umap", "pca"], key="reduction_method")
        
        # Adicionar campo para número de clusters quando a visualização por clusters for selecionada
        n_clusters = None
        if visualization_type == "Visualização por Clusters":
            n_clusters = st.slider("Número de clusters", min_value=2, max_value=15, value=6, key="n_clusters")
        
        btn_gerar = st.button("Gerar Visualização", key="generate_viz")
        
        if btn_gerar:
            if reduction_method == "- Selecione -":
                st.warning("⚠ Por favor, selecione um método de redução de dimensionalidade.")
            else:
                with st.spinner(f"Gerando visualização usando {reduction_method.upper()}... (pode levar alguns segundos)"):
                    if visualization_type == "Visualização Simples":
                        fig = plot_clean_embeddings(doc_embeddings, reduction_method)
                    else:  # Visualização por Clusters
                        fig = plot_grouped_embeddings(doc_names, doc_embeddings, reduction_method, n_clusters)
                        
                    if fig:
                        st.success("✅ Visualização gerada com sucesso!")
                        time.sleep(1)
                        st.pyplot(fig)
                    else:
                        st.error("❌ Não foi possível gerar a visualização.")


if __name__ == "__main__":
    run_dashboard()
