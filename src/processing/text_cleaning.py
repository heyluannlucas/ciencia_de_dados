import os
import re

# Caminhos dos arquivos
RAW_DOCS_FOLDER = "data/raw/docs/"
PROCESSED_DOCS_FOLDER = "data/processed/"

# Criar a pasta de saída caso não exista
os.makedirs(PROCESSED_DOCS_FOLDER, exist_ok=True)


def clean_text(text):
    """Remove espaços desnecessários e normaliza o texto"""
    text = re.sub(r"\s+", " ", text)  # Remove espaços extras
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove caracteres não ASCII
    text = re.sub(r"Navigation Menu.*?Explore All features", "", text, flags=re.DOTALL)  # Remove menus do GitHub
    text = text.strip()
    return text


def process_documents():
    """Processa todos os documentos baixados e aplica limpeza"""
    files = os.listdir(RAW_DOCS_FOLDER)
    total = len(files)

    if total == 0:
        print(f"Erro: Nenhum documento encontrado em {RAW_DOCS_FOLDER}. Execute download_docs.py primeiro.")
        return

    print(f"Iniciando processamento de {total} documentos...")

    for i, filename in enumerate(files):
        input_path = os.path.join(RAW_DOCS_FOLDER, filename)
        output_path = os.path.join(PROCESSED_DOCS_FOLDER, filename)

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned_text = clean_text(text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"[{i+1}/{total}] Processado: {filename}")
        except Exception as e:
            print(f"[{i+1}/{total}] Erro ao processar {filename}: {e}")

if __name__ == "__main__":
    process_documents()
