import datetime
import torch
from config import config_, load_graph_data
from gcn_model import ModelDeepNoFeatures  # Assuming you've renamed the class accordingly

##########################################################
EMBEDDING_SIZE = 512
NUM_EPOCHS = 1001
HIDDEN_LAYERS = 1024
LEARNING_RATE = 0.001
##########################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

chrom_ = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19', 'chrX'
]

for ch in chrom_:
    dir_ = f'/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/brain_without_common_graph/_{ch}'
    config_['graph_dir'] = dir_
    print(f'Working on {config_["graph_dir"]}, {datetime.datetime.now()}')

    graph_list = list(load_graph_data().values())

    # Assuming each graph_data object has a num_nodes attribute; adjust if necessary
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

            print(f'{datetime.datetime.now()}: Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')
            if epoch % 1000 == 0 and epoch != 0:
                torch.save(model, f'{config_["parent_dir"]}/{ch}_diff_loss_deep_model_no_features_{epoch}.pt')
                print(f'{ch} {epoch} saved')
