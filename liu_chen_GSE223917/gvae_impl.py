import random

import matplotlib.pyplot as plt
import torch
import umap
from sklearn.cluster import KMeans

from config import load_graph_data, config_
from gvae_model import GVAE, loss_function

#########################################################
NUM_EPOCHS = 10000
EMBEDDING_SIZE = 128
HIDDEN_LAYERS = 256
NUM_NODE_FEATURES = 1  # actually data.num_node_features
LEARNING_RATE = 0.001
NUM_CLUSTERS = 7
#########################################################

#########################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GVAE(in_features=NUM_NODE_FEATURES, hidden_dim=HIDDEN_LAYERS, latent_dim=EMBEDDING_SIZE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
graph_list = load_graph_data()

graph_list_ = list(graph_list.values())
#########################################################

#########################################################
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    random.shuffle(graph_list_)

    for graph in graph_list_:
        graph_data = graph.to(device)
        x, edge_index = graph_data.x, graph_data.edge_index
        model.train()
        optimizer.zero_grad()
        recon_graph, mu, logstd = model(x, edge_index)
        loss = loss_function(recon_graph, x, mu, logstd)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        graph_data = graph_data.to('cpu')
        del graph_data
    print(f'{epoch}th Epoch: Average Loss: {total_loss / len(graph_list_)}')
    if epoch % 100 == 0 and epoch != 0:
        torch.save(model, f'{config_["parent_dir"]}/GVAE_brain_chr1_model_{epoch}.pt')
        print(f'{epoch} saved')
        break

#########################################################
print("Extracting embeddings")
model.eval()
graph_embeddings = {}
for key, value in graph_list.items():
    graph = value.to(device)
    with torch.no_grad():
        mu, _ = model.encode(graph.x, graph.edge_index)
        graph_embedding = mu.mean(dim=0)  # Aggregate node embeddings to get a single graph embedding
        graph_embeddings[key] = graph_embedding.cpu().numpy()
        graph.to('cpu')

print("Reducing using UMAP")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer.fit_transform([graph_embeddings[i] for i in graph_embeddings])

print("Doing KMeans")
kmeans = KMeans(n_clusters=NUM_CLUSTERS)
clusters = kmeans.fit_predict(embedding_2d)
#########################################################

#########################################################
clustered_names = {}
for cluster, name in zip(clusters, graph_embeddings.keys()):
    if cluster in clustered_names:
        clustered_names[cluster].append(name)
    else:
        clustered_names[cluster] = [name]

print(clustered_names)
plt.figure(figsize=(12, 12))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('Graph Embeddings clustered with UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('plot_gvae.png')
print('Plotting done')
##########################################################
