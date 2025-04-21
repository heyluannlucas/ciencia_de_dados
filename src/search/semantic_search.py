import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Caminhos dos arquivos
EMBEDDINGS_FILE = "embeddings/document_embeddings.pkl"
PROCESSED_DOCS_FOLDER = "data/processed/"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Deve ser o mesmo do generate_embeddings.py
model = SentenceTransformer(MODEL_NAME)



def load_embeddings():
    """Carrega os embeddings armazenados e retorna nomes e vetores"""
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Erro: Arquivo {EMBEDDINGS_FILE} não encontrado. Execute generate_embeddings.py primeiro.")
        return None, None

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_dict = pickle.load(f)

    doc_names = list(embeddings_dict.keys())  # Lista com os nomes dos documentos
    doc_embeddings = np.array(list(embeddings_dict.values()))  # Matriz de embeddings

    return doc_names, doc_embeddings


def search(query, doc_names, doc_embeddings, top_n=5):
    """Realiza busca semântica nos documentos e retorna os mais relevantes"""
    if doc_names is None or doc_embeddings is None:
        print("Erro: Os embeddings não foram carregados corretamente.")
        return []

    # Gerar embedding da consulta
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Calcular similaridade de cosseno entre a consulta e os documentos
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Ordenar os resultados por relevância (maior similaridade primeiro)
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = [(doc_names[i], similarities[i]) for i in top_indices]

    return results


def display_document_content(doc_name):
    """Exibe o conteúdo do documento formatado"""
    doc_path = os.path.join(PROCESSED_DOCS_FOLDER, doc_name)

    if not os.path.exists(doc_path):
        print(f"❌ Erro: O arquivo {doc_name} não foi encontrado.")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"\n📄 **Conteúdo do documento: {doc_name}**\n")
    print(content[:1000])  # Exibe apenas os primeiros 1000 caracteres para evitar texto muito grande
    print("\n[... Documento truncado para visualização ...]\n")


if __name__ == "__main__":
    doc_names, doc_embeddings = load_embeddings()

    if doc_names is None or doc_embeddings is None:
        print("Erro: Não foi possível carregar os embeddings.")
    else:
        query = input("Digite sua busca: ")
        results = search(query, doc_names, doc_embeddings)

        print("\n📌 Resultados mais relevantes:\n")
        for doc, score in results:
            print(f"🔹 {doc} → Similaridade: {score:.4f}")

        # Perguntar ao usuário se deseja visualizar o conteúdo dos documentos
        while True:
            doc_choice = input(
                "\n📖 Digite o nome do documento para visualizar o conteúdo (ou 'sair' para encerrar): ").strip()
            if doc_choice.lower() == "sair":
                break
            display_document_content(doc_choice)
