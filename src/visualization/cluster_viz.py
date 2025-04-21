import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import umap

# Caminho do arquivo de embeddings
EMBEDDINGS_FILE = "embeddings/document_embeddings.pkl"

def load_embeddings():
    """Carrega os embeddings armazenados"""
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Erro: Arquivo {EMBEDDINGS_FILE} não encontrado. Execute generate_embeddings.py primeiro.")
        return None, None

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_dict = pickle.load(f)

    doc_names = list(embeddings_dict.keys())
    doc_embeddings = np.array(list(embeddings_dict.values()))

    return doc_names, doc_embeddings

def plot_clean_embeddings(doc_embeddings, reduction_method):
    """Reduz a dimensionalidade dos embeddings e plota os clusters
    
    Args:
        doc_names: Lista com os nomes dos documentos
        doc_embeddings: Array com os embeddings dos documentos
        reduction_method: Método de redução de dimensionalidade ("tsne", "umap" ou "pca")
        
    Returns:
        fig: Figura matplotlib com o gráfico gerado
    """

    if doc_embeddings is None or len(doc_embeddings) == 0:
        print("Erro: Nenhum embedding disponível para visualização.")
        return None

    print(f"Reduzindo dimensionalidade com {reduction_method.upper()}...")

    if reduction_method == "tsne":
        reduced_embeddings = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=1).fit_transform(
            doc_embeddings)

    elif reduction_method == "umap":
        reduced_embeddings = umap.UMAP(n_components=2, random_state=42).fit_transform(doc_embeddings)
    elif reduction_method == "pca":
        reduced_embeddings = PCA(n_components=2).fit_transform(doc_embeddings)
    else:
        print("Método desconhecido. Use 'tsne', 'umap' ou 'pca'.")
        return None

    # Criar o gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)


    plt.title(f"Visualização de Clusters ({reduction_method.upper()})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
    
    return fig

def plot_grouped_embeddings(doc_names, doc_embeddings, reduction_method, n_clusters=None):
    """Agrupa os documentos automaticamente por similaridade e plota os clusters
    
    Args:
        doc_names: Lista com os nomes dos documentos
        doc_embeddings: Array com os embeddings dos documentos
        reduction_method: Método de redução de dimensionalidade ("tsne", "umap" ou "pca")
        n_clusters: Número de clusters para agrupar os documentos. Se None, será otimizado

    Returns:
        fig: Figura matplotlib com o gráfico gerado
    """
    if doc_embeddings is None or len(doc_embeddings) == 0 or doc_names is None or len(doc_names) == 0:
        print("Erro: Nenhum embedding disponível para visualização.")
        return None
    
    # Aplicar K-means para determinar os clusters
    print(f"Agrupando documentos em {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(doc_embeddings)
    
    print(f"Reduzindo dimensionalidade com {reduction_method.upper()}...")
    
    # Primeiro passo: redução para dimensionalidade intermediária (10D)
    # Isso ajuda a preservar mais informações antes da projeção final 2D
    if doc_embeddings.shape[1] > 10:  # Se a dimensionalidade original for maior que 10
        pca_embed = PCA(n_components=10).fit_transform(doc_embeddings)
    else:
        pca_embed = doc_embeddings.copy()
    
    # Reduzir dimensionalidade com o método escolhido, com parâmetros ajustados para maximizar separação
    if reduction_method == "tsne":
        # Aumentar perplexity e early_exaggeration ajuda na separação dos clusters
        reduced_embeddings = TSNE(
            n_components=2, 
            perplexity=min(40, len(doc_embeddings)//5), 
            early_exaggeration=20,
            random_state=42, 
            n_jobs=-1
        ).fit_transform(pca_embed)
    elif reduction_method == "umap":
        # Aumentar n_neighbors amplia a visão global e minimal_dist força mais separação
        reduced_embeddings = umap.UMAP(
            n_components=2, 
            n_neighbors=min(30, len(doc_embeddings)//3),
            min_dist=0.3,
            random_state=42
        ).fit_transform(pca_embed)
    elif reduction_method == "pca":
        reduced_embeddings = PCA(n_components=2).fit_transform(pca_embed)
    else:
        print("Método desconhecido. Use 'tsne', 'umap' ou 'pca'.")
        return None
    
    # Expandir a visualização para evitar sobreposição
    # Multiplicamos as coordenadas por um fator para aumentar a separação
    scale_factor = 1.5
    reduced_embeddings *= scale_factor
    
    # Extrair palavras-chave das principais categorias
    cluster_keywords = {}
    for cluster_id in range(n_clusters):
        docs_in_cluster = [doc_names[i] for i in range(len(doc_names)) if cluster_labels[i] == cluster_id]
        
        # Extrair palavras-chave do nome dos documentos
        all_words = []
        for doc in docs_in_cluster:
            # Limpar o nome do arquivo
            clean_doc = doc.replace('.txt', '').replace('_', ' ')
            all_words.extend(clean_doc.split())
        
        # Encontrar as palavras mais frequentes (excluindo palavras comuns)
        word_freq = {}
        stop_words = {"and", "the", "to", "of", "in", "for", "on", "with", "by", "a", 
                      "an", "is", "it", "are", "this", "that", "there", "their"}
        for word in all_words:
            if word.lower() not in stop_words and len(word) > 2:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        
        # Selecionar as 3 palavras mais frequentes para representar o cluster
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_words:
            cluster_keywords[cluster_id] = ", ".join(word for word, _ in top_words)
        else:
            cluster_keywords[cluster_id] = f"Cluster {cluster_id+1}"
    
    # Criar figura com tamanho maior para melhor visualização
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Definir um mapa de cores com cores distintas
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # Calcular centroides de cada cluster para uso posterior
    centroids = []
    for i in range(n_clusters):
        points = reduced_embeddings[cluster_labels == i]
        if len(points) > 0:
            centroids.append(np.mean(points, axis=0))
    
    # Desenhar polígonos convexos para cada cluster com menor transparência (alpha)
    # e com margens entre clusters para melhor separação visual
    for i in range(n_clusters):
        # Filtrar pontos deste cluster
        cluster_points = reduced_embeddings[cluster_labels == i]
        
        if len(cluster_points) < 3:  # Precisamos de pelo menos 3 pontos para um polígono
            continue
        
        try:
            # Criar o Convex Hull
            hull = ConvexHull(cluster_points)
            
            # Reduzir levemente o tamanho do hull para criar espaço entre clusters
            centroid = np.mean(cluster_points, axis=0)
            shrink_factor = 1  # Reduz o hull em 5% em direção ao centroide
            
            # Desenhar o polígono do hull com contorno mais definido
            for simplex in hull.simplices:
                hull_points = cluster_points[simplex]
                # Encolher o polígono em direção ao centroide para criar espaço entre clusters
                shrunk_points = centroid + shrink_factor * (hull_points - centroid)
                
                plt.fill(
                    shrunk_points[:, 0], 
                    shrunk_points[:, 1], 
                    alpha=0.25,  # Menor opacidade para melhor visibilidade
                    color=colors[i % len(colors)],
                    edgecolor=colors[i % len(colors)],
                    linewidth=2
                )
        except:
            # Se não for possível criar um hull (pontos colineares)
            pass
    
    # Plotar os pontos por cima dos polígonos
    scatter = ax.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=[colors[label % len(colors)] for label in cluster_labels], 
        s=60,  # Tamanho maior para pontos mais visíveis
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Adicionar anotações para os clusters em caixas mais destacadas
    for i in range(n_clusters):
        if i < len(centroids):
            # Definir tamanho do texto baseado no número de documentos no cluster
            docs_count = np.sum(cluster_labels == i)
            font_size = min(14, 8 + (docs_count / len(doc_names)) * 8)
            
            ax.annotate(
                cluster_keywords[i],
                (centroids[i][0], centroids[i][1]),
                fontsize=font_size,
                ha='center',
                va='center',
                weight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc="white",
                    ec=colors[i % len(colors)],
                    lw=2,
                    alpha=0.9
                )
            )
    
    plt.title(f"Visualização de Clusters por Similaridade ({reduction_method.upper()})", fontsize=16)
    
    # Remover eixos para uma visualização mais limpa
    plt.axis('off')
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Carregar embeddings antes de chamar a visualização
    doc_names, doc_embeddings = load_embeddings()

    if doc_names is None or doc_embeddings is None:
        print("⚠ Erro: Não foi possível carregar os embeddings.")
    else:
        # Permitir escolher o tipo de visualização
        print("\nEscolha o tipo de visualização:")
        print("1. Visualização Simples")
        print("2. Visualização por Clusters")
        viz_choice = input("Digite 1 ou 2: ").strip()
        
        method = input("\nEscolha o método de redução de dimensionalidade (tsne, umap, pca): ").strip().lower()
        
        if viz_choice == "1":
            fig = plot_clean_embeddings(doc_embeddings, method)
        else:
            n_clusters = input("\nNúmero de clusters (deixe em branco para otimização automática): ").strip()
            n_clusters = int(n_clusters) if n_clusters else None
            fig = plot_grouped_embeddings(doc_names, doc_embeddings, method, n_clusters)
            
        if fig:
            plt.show()
