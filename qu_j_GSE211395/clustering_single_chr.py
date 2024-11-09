import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import load_graph_data, calculate_score

chrom_ = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19',
    # 'chr20', 'chr21', 'chr22',  #mouse doesn't have these
    'chrX'
]
main_cluster_names = {}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_scores = {}

for ch in chrom_:
    graph_dir = f'/mmfs1/scratch/utsha.saha/mouse_data/data/datasets/qu_j_GSE211395_serum_2i/graphs/_{ch}'
    model_dir = f"/mmfs1/scratch/utsha.saha/mouse_data/data/datasets/qu_j_GSE211395_serum_2i/models/{ch}_no_features_100.pt"
    graph_list = load_graph_data(graph_dir)

    print("=======================================================================")
    print(f"============================={ch}=====================================")
    print(f'Working on {graph_dir}')
    print(f"model: {model_dir}")

    model = torch.load(model_dir, map_location='cuda:0')
    model.eval()

    print("Extracting embeddings")
    graph_embeddings = {}

    with torch.no_grad():
        for key, value in graph_list.items():
            graph_data = value.to(device)
            node_embeddings = model(graph_data)  # Get the embedding for the graph
            graph_embedding = node_embeddings.sum(dim=0)
            graph_embeddings[key] = graph_embedding.cpu().numpy()
            graph_data.to('cpu')
            del graph_data

    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])

    for x in range(2, 8):
        print(f"============================={x}=====================================")
        plot_title_name = f'GSE239969 without RNA, {ch}, {x} clusters '
        plot_file_name = f'plots/GSE211395_no_rna_{ch}_{x}.png'

        kmeans = KMeans(n_clusters=x)  # Set the number of clusters
        predicted_labels = kmeans.fit_predict(embedding_2d)

        clustered_names = {}
        for cluster, name in zip(predicted_labels, graph_embeddings.keys()):
            if cluster in clustered_names:
                clustered_names[cluster].append(name)
            else:
                clustered_names[cluster] = [name]

        # print(clustered_names)
        main_cluster_names[ch] = clustered_names

        fig, ax = plt.subplots(figsize=(12, 12))
        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=predicted_labels, cmap='Spectral', s=50, alpha=0.6, label='Clusters')  # Use ax.scatter instead of plt.scatter
        ax.set_title(plot_title_name, fontsize=16)
        ax.set_xlabel('UMAP Dimension 1', fontsize=13)
        ax.set_ylabel('UMAP Dimension 2', fontsize=13)

        # Adjust colorbar size
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.4, fraction=0.046, pad=0.04, label='Cluster ID')  # Adjust fraction and pad to control the colorbar size and spacing

        # Create and position legend
        unique_labels = np.unique(predicted_labels)
        colors = [scatter.cmap(scatter.norm(label)) for label in unique_labels]
        custom_legends = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.6) for color in colors]
        ax.legend(custom_legends, [f'Cluster {label}' for label in unique_labels], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(True)
        fig.tight_layout()  # Adjust layout to accommodate the main plot, legend, and colorbar
        fig.savefig(plot_file_name, dpi=300)  # Save the plot with high resolution
        print('Enhanced plotting with legend and smaller colorbar done')

        labels_pred = list(predicted_labels)
        labels_true = [i.split("_")[1] + i.split("_")[2] for i in graph_list.keys()]

        scores = calculate_score(labels_true, labels_pred)
        silhouette_avg = silhouette_score(embedding_2d, predicted_labels)
        scores.update({"silhouette_avg": silhouette_avg})
        print(f'Silhouette Score for: {silhouette_avg}')
        all_scores[f'{ch}_{x}'] = scores

        print("====================================================================")
    print("====================================================================")

print(all_scores)
