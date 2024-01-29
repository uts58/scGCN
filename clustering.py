import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import load_graph_data, config_

true_cluster_dict = pd.read_csv(config_['labels']).set_index('cell_name').to_dict()['cell_type']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(f'{config_["parent_dir"]}/brain_model.pt')
model.eval()  # Set the model to evaluation mode

graph_list = load_graph_data()
graph_embeddings = {}

with torch.no_grad():
    for key, value in graph_list.items():
        graph_data = value.to(device)
        node_embeddings = model(graph_data)  # Get the embedding for the graph
        graph_embedding = node_embeddings.mean(dim=0)
        graph_embeddings[key] = graph_embedding.cpu().numpy()
        graph_data.to('cpu')

print(len(graph_embeddings), len(true_cluster_dict))

keys_to_remove = [key for key in true_cluster_dict if key not in graph_embeddings.keys()]
print(f'Removing {keys_to_remove}')
for key in keys_to_remove:
    true_cluster_dict.pop(key)

keys_to_remove = [key for key in graph_embeddings.keys() if key not in true_cluster_dict]
print(f'Removing {keys_to_remove}')
for key in keys_to_remove:
    graph_embeddings.pop(key)

graph_names = list(graph_embeddings.keys())

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])

kmeans = KMeans(n_clusters=7)  # Set the number of clusters
predicted_labels = kmeans.fit_predict(embedding_2d)

print('Calculating Adjusted Rand Index')
true_labels = [true_cluster_dict[name] for name in graph_names]
print("True labels========================================")
print(true_labels)
print("Predicted labels===================================")
print(predicted_labels)
ari_score = adjusted_rand_score(true_labels, predicted_labels)
print(f"Adjusted Rand Index: {ari_score}")

# Plotting
plt.figure(figsize=(12, 12))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=predicted_labels, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('Graph Embeddings clustered with UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('plot.png')
print('Plotting done')
