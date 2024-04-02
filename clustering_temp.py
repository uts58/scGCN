import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score

from config import load_graph_data, config_, calculate_score

df_temp = pd.read_csv(config_['labels'])
cell_type_dict = dict(zip(df_temp['cell_name'], df_temp['cell_type']))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("/mmfs1/scratch/utsha.saha/mouse_data/data/models/brain_without_common_graph_deep_model_1000.pt")
model.eval()  # Set the model to evaluation mode

config_['graph_dir'] = "/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/brain_without_common_graph/"

graph_list = load_graph_data()
for items in graph_list.copy():
    if items not in cell_type_dict:
        graph_list.pop(items)
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

    for x in range(7, 15):
        print(f"============================={x}=====================================")
        kmeans = KMeans(n_clusters=x)  # Set the number of clusters
        predicted_labels = kmeans.fit_predict(embedding_2d)

        clustered_names = {}
        for cluster, name in zip(predicted_labels, graph_embeddings.keys()):
            if cluster in clustered_names:
                clustered_names[cluster].append(name)
            else:
                clustered_names[cluster] = [name]

        fig, ax = plt.subplots(figsize=(12, 12))
        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=predicted_labels, cmap='Spectral', s=50, alpha=0.6, label='Clusters')  # Use ax.scatter instead of plt.scatter
        ax.set_title('Graph Embeddings clustered with UMAP and K-Means', fontsize=18)
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)

        # Adjust colorbar size
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, fraction=0.046, pad=0.04, label='Cluster ID')  # Adjust fraction and pad to control the colorbar size and spacing

        # Create and position legend
        unique_labels = np.unique(predicted_labels)
        colors = [scatter.cmap(scatter.norm(label)) for label in unique_labels]
        custom_legends = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.6) for color in colors]
        ax.legend(custom_legends, [f'Cluster {label}' for label in unique_labels], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(True)
        fig.tight_layout()  # Adjust layout to accommodate the main plot, legend, and colorbar
        fig.savefig(f'cluster_{x}_plot.png', dpi=300)  # Save the plot with high resolution
        print('Enhanced plotting with legend and smaller colorbar done')

        labels_pred = list(predicted_labels)
        labels_true = [cell_type_dict[cell_name] for cell_name in graph_list.keys()]

        calculate_score(labels_true, labels_pred)
        silhouette_avg = silhouette_score(embedding_2d, predicted_labels)
        print(f'Silhouette Score for: {silhouette_avg}')

        print("====================================================================")
    print("====================================================================")

    print(clustered_names)
