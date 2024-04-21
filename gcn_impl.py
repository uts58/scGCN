import datetime
import random

import torch

from config import config_, load_graph_data
from gcn_model import ModelDeep

##########################################################
EMBEDDING_SIZE = 512
NUM_EPOCHS = 1001
HIDDEN_LAYERS = 1024
NUM_NODE_FEATURES = 1  # actually data.num_node_features
LEARNING_RATE = 0.001
##########################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelDeep(num_node_features=NUM_NODE_FEATURES, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

graph_dir = '/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/embroy_without_common_graph/'

print(f'Graph dir: {graph_dir}')
graph_list = list(load_graph_data(graph_dir).values())

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    random.shuffle(graph_list)

    for graph in graph_list:
        graph_data = graph.to(device)
        model.train()
        optimizer.zero_grad()
        embeddings = model.forward(graph_data)
        loss = torch.var(embeddings)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        graph_data = graph_data.to('cpu')
        del graph_data

    print(f'{datetime.datetime.now()}: Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')
    if epoch % 1000 == 0 and epoch != 0:
        torch.save(model, f'{config_["parent_dir"]}/embroy_all_chr_{epoch}.pt')
        print(f'{epoch} saved')
