from sentence_transformers import SentenceTransformer
import os
import pickle
import numpy as np

# Caminhos
PROCESSED_DOCS_FOLDER = "data/processed/"
EMBEDDINGS_FILE = "embeddings/document_embeddings.pkl"

# Modelo de Embeddings (ATUALIZADO)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)


def generate_embeddings():
    """Gera embeddings para os documentos processados e salva no arquivo"""
    doc_embeddings = {}

    if not os.path.exists(PROCESSED_DOCS_FOLDER):
        print("❌ A pasta de documentos processados não foi encontrada.")
        return
    
    for filename in os.listdir(PROCESSED_DOCS_FOLDER):
        file_path = os.path.join(PROCESSED_DOCS_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Criar embedding do conteúdo
        embedding = model.encode(content, convert_to_numpy=True)
        doc_embeddings[filename] = embedding

    # Salvar embeddings
    os.makedirs("embeddings", exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(doc_embeddings, f)

    print(f"✅ {len(doc_embeddings)} embeddings gerados e salvos em {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    generate_embeddings()
