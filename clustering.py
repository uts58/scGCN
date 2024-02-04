import matplotlib.pyplot as plt
import torch
import umap
from sklearn.cluster import KMeans

from config import load_graph_data, config_

# true_cluster_dict = pd.read_csv(config_['labels']).set_index('cell_name').to_dict()['cell_type']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(f'{config_["parent_dir"]}/GVAE_brain_model_100.pt')
model.eval()

graph_list = load_graph_data()

graph_embeddings = {}

with torch.no_grad():
    for key, value in graph_list.items():
        graph_data = value.to(device)
        #FOR GCN
        # node_embeddings = model(graph_data)  # Get the embedding for the graph
        ###################
        #FOR GVAE
        x, edge_index = graph_data.x, graph_data.edge_index
        node_embeddings = model(x, edge_index)
        ###################
        graph_embedding = node_embeddings.sum(dim=0)
        graph_embeddings[key] = graph_embedding.cpu().numpy()
        graph_data.to('cpu')

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])

kmeans = KMeans(n_clusters=7)  # Set the number of clusters
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

print(clustered_names)
plt.figure(figsize=(12, 12))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=predicted_labels, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('Graph Embeddings clustered with UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('plot_gvae.png')
print('Plotting done')
