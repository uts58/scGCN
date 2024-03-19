import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

from config import load_graph_data, config_

# true_cluster_dict = pd.read_csv(config_['labels']).set_index('cell_name').to_dict()['cell_type']


chrom_ = [
    'chr1',
    'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19',
    # 'chr20', 'chr21', 'chr22',  #mouse doesn't have these
    'chrX'
]
main_cluster_names = {}

for ch in chrom_:
    dir_ = f'/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/brain_without_common_graph/_{ch}'
    config_['graph_dir'] = dir_
    graph_list = load_graph_data()

    print(f'Working on {config_["graph_dir"]}, {datetime.datetime.now()}, {config_["parent_dir"]}/{ch}_diff_loss_deep_model_1000.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f'{config_["parent_dir"]}/{ch}_diff_loss_deep_model_1000.pt')
    model.eval()

    print("Extracting embeddings")
    graph_embeddings = {}

    with torch.no_grad():
        for key, value in graph_list.items():
            graph_data = value.to(device)
            # FOR GCN
            node_embeddings = model(graph_data)  # Get the embedding for the graph
            graph_embedding = node_embeddings.sum(dim=0)
            ###################
            # FOR GVAE
            # x, edge_index = graph_data.x, graph_data.edge_index
            # recon_graph, mu, logstd = model(x, edge_index)
            # graph_embedding = mu.sum(dim=0)
            ###################
            graph_embeddings[key] = graph_embedding.cpu().numpy()
            graph_data.to('cpu')

    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])

    kmeans = KMeans(n_clusters=4)  # Set the number of clusters
    predicted_labels = kmeans.fit_predict(embedding_2d)

    # print(predicted_labels)
    # print('=================================')
    # print([i for i in graph_embeddings.keys()])

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
    fig.savefig(f'{ch}_mean_diff_loss_plot.png', dpi=300)  # Save the plot with high resolution
    print('Enhanced plotting with legend and smaller colorbar done')

print(main_cluster_names)
