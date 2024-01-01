import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from config import load_graph_data, chunk_graphs
from new_model import GVAE, loss_function

#########################################################
NUM_EPOCHS = 50
NUM_CLUSTERS = 5

BATCH_SIZE = 500
EMBEDDING_SIZE = 10
HIDDEN_LAYERS = 16
NUM_NODE_FEATURES = 1  # actually data.num_node_features
#########################################################

#########################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GVAE(in_features=NUM_NODE_FEATURES, hidden_dim=HIDDEN_LAYERS, latent_dim=EMBEDDING_SIZE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
graph_list = load_graph_data()

data_loader = DataLoader(chunk_graphs(graph_list, BATCH_SIZE), shuffle=True)

#########################################################
for epoch in range(NUM_EPOCHS):
    for graph in graph_list:
        x, edge_index = graph.x, graph.edge_index
        model.train()
        optimizer.zero_grad()
        recon_graph, mu, logstd = model(x, edge_index)
        loss = loss_function(recon_graph, x, mu, logstd)
        loss.backward()
        optimizer.step()

#########################################################
graph_embeddings = []
for graph in graph_list:
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(graph.x, graph.edge_index)
        graph_embedding = mu.mean(dim=0)  # Aggregate node embeddings to get a single graph embedding
        graph_embeddings.append(graph_embedding.numpy())

graph_embeddings = np.stack(graph_embeddings)
# graph_embeddings = np.vstack(graph_embeddings)
kmeans = KMeans(n_clusters=NUM_CLUSTERS)
clusters = kmeans.fit_predict(graph_embeddings)
#########################################################
