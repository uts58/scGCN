import random

import torch

from config import load_graph_data, config_
from gvae_model import GVAE, loss_function

#########################################################
NUM_EPOCHS = 10000
EMBEDDING_SIZE = 128
HIDDEN_LAYERS = 256
NUM_NODE_FEATURES = 1  # actually data.num_node_features
LEARNING_RATE = 0.001
#########################################################

#########################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GVAE(in_features=NUM_NODE_FEATURES, hidden_dim=HIDDEN_LAYERS, latent_dim=EMBEDDING_SIZE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
graph_list = list(load_graph_data().values())

# data_loader = DataLoader(chunk_graphs(graph_list, BATCH_SIZE), shuffle=True)

#########################################################
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    random.shuffle(graph_list)

    for graph in graph_list:
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
    print(f'{epoch}th Epoch: Average Loss: {total_loss / len(graph_list)}')
    if epoch % 100 == 0:
        torch.save(model, f'{config_["parent_dir"]}/GVAE_brain_model_{epoch}.pt')
        print(f'{epoch} saved')

#########################################################
# graph_embeddings = []
# for graph in graph_list:
#     model.eval()
#     with torch.no_grad():
#         mu, _ = model.encode(graph.x, graph.edge_index)
#         graph_embedding = mu.mean(dim=0)  # Aggregate node embeddings to get a single graph embedding
#         graph_embeddings.append(graph_embedding.numpy())
#
# graph_embeddings = np.stack(graph_embeddings)
# # graph_embeddings = np.vstack(graph_embeddings)
# kmeans = KMeans(n_clusters=NUM_CLUSTERS)
# clusters = kmeans.fit_predict(graph_embeddings)
#########################################################
