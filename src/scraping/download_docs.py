import os
import json
import requests
from bs4 import BeautifulSoup

# Caminhos dos arquivos
LINKS_FILE = "data/raw/awesome_links.json"
OUTPUT_FOLDER = "data/raw/docs/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):
    """Remove espaços desnecessários e normaliza o texto"""
    return " ".join(text.split())


def download_documentation():
    """Faz o download do conteúdo dos links extraídos"""
    if not os.path.exists(LINKS_FILE):
        print(f"Erro: Arquivo {LINKS_FILE} não encontrado. Execute extract_links.py primeiro.")
        return

    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        links = json.load(f)

    total = len(links)
    print(f"Iniciando download de {total} documentações...")

    for i, link in enumerate(links):
        url = link["url"]
        title = link["title"].replace(" ", "_").lower()
        file_path = os.path.join(OUTPUT_FOLDER, f"{title}.txt")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Extrair o texto da página HTML
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            text = clean_text(text)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[{i + 1}/{total}] Sucesso: {title}")
        except Exception as e:
            print(f"[{i + 1}/{total}] Erro ao baixar {url}: {e}")


if __name__ == "__main__":
    download_documentation()
