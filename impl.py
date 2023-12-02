import glob
import pickle

import matplotlib.pyplot as plt
import torch
import umap
from sklearn.cluster import KMeans

from config import config_
from models import GCN

EMBEDDING_SIZE = 10
NUM_EPOCHS = 50
HIDDEN_LAYERS = 16
NUM_NODE_FEATURES = 1  # actually data.num_node_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=NUM_NODE_FEATURES, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f'Graph dir: {config_["graph_dir"]}')

graph_data_list = []
graph_names = []
for files in glob.glob(f'{config_["graph_dir"]}/*.pkl'):
    graph_data_list.append(pickle.load(open(files, 'rb')))
    graph_names.append(files.split('/')[-1].replace('pkl', ''))

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i, items in enumerate(graph_data_list):
        # print(f'Training {graph_names[i]}')
        graph_data = items.to(device)
        graph_data.x = graph_data.x.float()
        graph_data.edge_index = graph_data.edge_index.int()

        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(graph_data)

        # Assuming a simple reconstruction loss: MSE between input features and embeddings
        loss = torch.var(embeddings)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f'Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_data_list)}')

print('Training done')
torch.save(model, f'{config_["graph_dir"]}/model.pt')

graph_embeddings = []
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for items in graph_data_list:
        graph_data = items.to(device)
        node_embeddings = model(graph_data)  # Get the embedding for the graph
        graph_embedding = node_embeddings.mean(dim=0)
        graph_embeddings.append(graph_embedding.cpu().numpy())

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer.fit_transform(graph_embeddings)

kmeans = KMeans(n_clusters=10)  # Set the number of clusters
clusters = kmeans.fit_predict(embedding_2d)

# Plotting
plt.figure(figsize=(12, 12))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('Graph Embeddings clustered with UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('plot.png')
