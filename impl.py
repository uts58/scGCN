import torch

from config import config_, load_graph_data
from models import GCN

##########################################################
EMBEDDING_SIZE = 128
NUM_EPOCHS = 50
HIDDEN_LAYERS = 256
NUM_NODE_FEATURES = 1  # actually data.num_node_features
##########################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=NUM_NODE_FEATURES, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print(f'Graph dir: {config_["graph_dir"]}')
graph_list = load_graph_data()

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for key, value in graph_list.items():
        # print(f'Training {key}')
        graph_data = value.to(device)

        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model.forward(graph_data)

        # Reconstruction loss: MSE between input features and embeddings
        # loss = torch.mean((embeddings - graph_data.x) ** 2)
        loss = torch.var(embeddings)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        graph_data = graph_data.to('cpu')
        del graph_data

    print(f'Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')

print('Training done')
torch.save(model, f'{config_["parent_dir"]}/brain_model.pt')
