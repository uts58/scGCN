import torch

from config import config_, load_graph_data
from gcn_model import ModelDeep

##########################################################
EMBEDDING_SIZE = 256
NUM_EPOCHS = 100000
HIDDEN_LAYERS = 512
NUM_NODE_FEATURES = 1  # actually data.num_node_features
LEARNING_RATE = 0.001
##########################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(num_node_features=NUM_NODE_FEATURES, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS)
model = ModelDeep(num_node_features=NUM_NODE_FEATURES, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f'Graph dir: {config_["graph_dir"]}')
graph_list = list(load_graph_data().values())

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for graph in graph_list:
        graph_data = graph.to(device)
        model.train()
        optimizer.zero_grad()
        embeddings = model.forward(graph_data)
        loss = torch.nn.functional.mse_loss(embeddings, graph_data.x)
        # loss = torch.mean((embeddings - graph_data.x) ** 2)
        # loss = torch.var(embeddings)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        graph_data = graph_data.to('cpu')
        del graph_data

    print(f'Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')
    if epoch % 100 == 0 and epoch != 0:
        torch.save(model, f'{config_["parent_dir"]}/deep_model{epoch}.pt')
        print(f'{epoch} saved')
