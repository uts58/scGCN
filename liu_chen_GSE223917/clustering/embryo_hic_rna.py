from itertools import product

import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import load_graph_data, calculate_score

# Load cell type dictionary
df_temp = pd.read_csv("/mmfs1/scratch/utsha.saha/mouse_data/data/datasets/liu_chen_GSE223917_brain_embryo/labels_embryo.csv")
cell_type_dict = dict(zip(df_temp['cell_name'], df_temp['cell_type']))

chrom_ = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19', 'chrX'
]

main_cluster_names = {}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# UMAP grid search parameters
n_neighbors_list = [5, 10, 15]
min_dist_list = [0.1, 0.5, 0.9]
n_components_list = [2, 16, 32, 64, 128, 256]
n_clusters_for_kmaens = 21


# Function to process each chromosome
def process_chromosome(ch):
    graph_dir = f'/mmfs1/scratch/utsha.saha/mouse_data/data/datasets/liu_chen_GSE223917_brain_embryo/graphs/embryo_without_common_graph/_{ch}'
    model_dir = f"/mmfs1/scratch/utsha.saha/mouse_data/data/datasets/liu_chen_GSE223917_brain_embryo/models/embryo/{ch}_1000.pt"

    graph_list = load_graph_data(graph_dir)

    for items in graph_list.copy():
        if items not in cell_type_dict:
            graph_list.pop(items)

    print(f"============================={ch}=====================================")
    print(f'Working on {graph_dir}')
    print(f"model: {model_dir}")

    model = torch.load(model_dir, map_location=device)
    model.eval()
    print("Extracting embeddings")
    graph_embeddings = {}

    with torch.no_grad():
        for key, value in graph_list.items():
            graph_data = value.to(device)
            node_embeddings = model(graph_data)  # Get the embedding for the graph
            graph_embedding = node_embeddings.mean(dim=0)
            graph_embeddings[key] = graph_embedding.cpu().numpy()
            graph_data.to('cpu')

    best_params = []

    # Grid search on UMAP parameters
    for n_neighbors, min_dist, n_components in product(n_neighbors_list, min_dist_list, n_components_list):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
        embedding = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])
        kmeans = KMeans(n_init='auto', n_clusters=n_clusters_for_kmaens)

        predicted_labels = kmeans.fit_predict(embedding)

        clustered_names = {}
        for cluster, name in zip(predicted_labels, graph_embeddings.keys()):
            if cluster in clustered_names:
                clustered_names[cluster].append(name)
            else:
                clustered_names[cluster] = [name]

        main_cluster_names[ch] = clustered_names

        # Calculate silhouette score
        labels_pred = list(predicted_labels)
        labels_true = [cell_type_dict[cell_name] for cell_name in graph_list.keys()]
        other_score = calculate_score(labels_true, labels_pred)
        silhouette_avg = silhouette_score(embedding, predicted_labels)
        print(f'Silhouette Score for n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}, silhouette_avg={silhouette_avg}')

        best_params.append([ch, n_neighbors, min_dist, n_components, silhouette_avg] + list(other_score.values()))

    return best_params


all_params = []
# Run the grid search in parallel across chromosomes using ThreadPoolExecutor
for ch in chrom_:
    best_params_ = process_chromosome(ch)
    all_params += best_params_

# Save results to CSV
key_list = [
    'chr', 'n_neighbors', 'min_dist', 'n_components', 'silhouette_score',
    'adjusted_rand', 'adjusted_mutual_info', 'completeness', 'fowlkes_mallows',
    'homogeneity', 'mutual_info', 'normalized_mutual_info', 'v_measure', 'rand'
]

df = pd.DataFrame(all_params, columns=key_list)
df.to_csv(f"embryo_hic_rna.csv", index=False)

