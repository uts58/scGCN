import datetime
import random
import threading

import torch

from config import config_, load_graph_data
from gcn_model import ModelDeepNoFeatures

##########################################################
EMBEDDING_SIZE = 512
NUM_EPOCHS = 101
HIDDEN_LAYERS = 1024
LEARNING_RATE = 0.001
##########################################################


chrom_ = [
    'chr1',
    'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19',
    # 'chr20', 'chr21', 'chr22',  #mouse doesn't have these
    'chrX'
]
graph_dir = "/mmfs1/scratch/utsha.saha/mouse_data/data/not_using/qu_j_GSE211395_serum_2i/graphs"


for ch in chrom:
    dir_ = f"{graph_dir}/_{ch}/"
    print(f'Working on {dir_}, {datetime.datetime.now()}, cuda:{no}')
    graph_list = list(load_graph_data(dir_).values())

    num_nodes = graph_list[0].num_nodes

    device = torch.device(f'cuda:{no}' if torch.cuda.is_available() else 'cpu')
    model = ModelDeepNoFeatures(num_nodes=num_nodes, embedding_size=EMBEDDING_SIZE, hidden_layers=HIDDEN_LAYERS)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        random.shuffle(graph_list)

        total_loss = 0
        for graph in graph_list:
            graph_data = graph.to(device)
            model.train()
            optimizer.zero_grad()
            embeddings = model.forward(graph_data)
            loss = torch.var(embeddings)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            del graph_data

        print(f'{no}: Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')
        if epoch % 100 == 0 and epoch != 0:
            torch.save(model, f'{config_["parent_dir"]}/models/{ch}_no_features_{epoch}.pt')
            print(f'{no}: {ch} {epoch} saved')


