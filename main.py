import os
import sys
import time
import subprocess
from src.scraping.extract_links import extract_links, save_links
from src.scraping.download_docs import download_documentation
from src.processing.text_cleaning import process_documents
from src.processing.generate_embeddings import generate_embeddings

LINKS_FILE = "data/raw/awesome_links.json"
RAW_DOCS_FOLDER = "data/raw/docs/"
PROCESSED_DOCS_FOLDER = "data/processed/"
EMBEDDINGS_FILE = "embeddings/document_embeddings.pkl"

def run_pipeline():
    """Executa as etapas do pipeline antes de iniciar o dashboard"""
    print("\n🚀 Iniciando o pipeline de processamento de documentação técnica...\n")

    if not os.path.exists(LINKS_FILE):
        print("🔗 Extraindo links das documentações...")
        links = extract_links()
        if links:
            save_links(links)
            print(f"✅ {len(links)} links extraídos com sucesso!")
        else:
            print("⚠ Nenhum link foi extraído.")
            return
    else:
        print("✅ Links já extraídos. Pulando esta etapa.")

    time.sleep(1)

    if not os.listdir(RAW_DOCS_FOLDER):
        print("\n📥 Baixando documentações...")
        download_documentation()
    else:
        print("✅ Documentações já baixadas. Pulando esta etapa.")

    time.sleep(1)

    if not os.listdir(PROCESSED_DOCS_FOLDER):
        print("\n🧹 Limpando e estruturando os textos...")
        process_documents()
    else:
        print("✅ Textos já processados. Pulando esta etapa.")

    time.sleep(1)

    if not os.path.exists(EMBEDDINGS_FILE):
        print("\n🧠 Gerando embeddings para os documentos...")
        generate_embeddings()
    else:
        print("✅ Embeddings já gerados. Pulando esta etapa.")

    print("\n✅ Pipeline concluído com sucesso!")

def start_dashboard():
    """Inicia o dashboard do Streamlit corretamente"""
    print("\n🌐 Iniciando o Dashboard do Streamlit...\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py", "--server.fileWatcherType", "none"])

if __name__ == "__main__":
    run_pipeline()
    start_dashboard()
