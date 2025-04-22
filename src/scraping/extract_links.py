import requests
from bs4 import BeautifulSoup
import json

URL = "https://raw.githubusercontent.com/sindresorhus/awesome/main/readme.md"


def extract_links():
    """Extrai links da documentação listada no repositório Awesome"""
    response = requests.get(URL)

    if response.status_code != 200:
        print(f"Erro ao acessar o repositório: {response.status_code}")
        return []

    content = response.text
    soup = BeautifulSoup(content, "html.parser")

    links = []
    for line in content.split("\n"):
        if line.startswith("- ["):
            try:
                parts = line.split("](")
                title = parts[0][3:].strip()
                url = parts[1].split(")")[0].strip()
                links.append({"title": title, "url": url})
            except IndexError:
                continue

    return links


def save_links(links, file_path="data/raw/awesome_links.json"):
    """Salva os links extraídos em um arquivo JSON"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=4, ensure_ascii=False)
    print(f"Links salvos em {file_path}")


if __name__ == "__main__":
    links = extract_links()
    if links:
        save_links(links)
        print(f"Extração concluída! Total de links extraídos: {len(links)}")
    else:
        print("Nenhum link foi extraído.")
