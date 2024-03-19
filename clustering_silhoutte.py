import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # Import silhouette_score

from config import load_graph_data, config_

chrom_ = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19', 'chrX'
    # Note: 'chr20', 'chr21', 'chr22' are commented out as mouse doesn't have these
]

for ch in chrom_:
    dir_ = f'/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/brain_without_common_graph/_{ch}'
    config_['graph_dir'] = dir_
    print(f'Working on {config_["graph_dir"]}, {datetime.datetime.now()}, {config_["parent_dir"]}/{ch}_deep_model_1000.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f'{config_["parent_dir"]}/{ch}_deep_model_1000.pt')
    model.eval()

    graph_list = load_graph_data()

    print("Extracting embeddings")
    graph_embeddings = {}

    with torch.no_grad():
        for key, value in graph_list.items():
            graph_data = value.to(device)
            node_embeddings = model(graph_data)  # Get the embedding for the graph
            graph_embedding = node_embeddings.sum(dim=0)
            graph_embeddings[key] = graph_embedding.cpu().numpy()
            graph_data.to('cpu')

    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])

    kmeans = KMeans(n_clusters=7)  # Specify the number of clusters
    predicted_labels = kmeans.fit_predict(embedding_2d)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(embedding_2d, predicted_labels)
    print(f'Silhouette Score for {ch}: {silhouette_avg}')

    # The rest of your plotting code remains the same
