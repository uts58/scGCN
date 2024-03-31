import torch

from config import config_, load_graph_data
from gcn_model import ModelDeepNoFeatures

##########################################################
EMBEDDING_SIZE = 512
NUM_EPOCHS = 1001
HIDDEN_LAYERS = 1024
LEARNING_RATE = 0.001
##########################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Graph dir: {config_["graph_dir"]}')
graph_list = list(load_graph_data().values())

if graph_list:
    num_nodes = graph_list[0].num_nodes  # This assumes all graphs have the same number of nodes
    model = ModelDeepNoFeatures(num_nodes=num_nodes, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f'Graph dir: {config_["graph_dir"]}')

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for graph in graph_list:
            graph_data = graph.to(device)
            model.train()
            optimizer.zero_grad()
            embeddings = model(graph_data)  # Using the model as a callable is preferred
            loss = torch.var(embeddings)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            graph_data = graph.to('cpu')
            del graph_data

        print(f'Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')
        if epoch % 1000 == 0 and epoch != 0:
            torch.save(model, f'{config_["parent_dir"]}/deep_model_no_features_with_common_graph_{epoch}.pt')
            print(f'{epoch} saved')
