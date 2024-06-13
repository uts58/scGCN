import datetime
import random
import threading

import torch

from config import config_, load_graph_data
from gcn_model import ModelDeep, ModelDeepNoFeatures

##########################################################
EMBEDDING_SIZE = 512
NUM_EPOCHS = 1001
HIDDEN_LAYERS = 1024
LEARNING_RATE = 0.001
##########################################################


chrom_ = ['chr16', 'chr17', 'chr18', 'chr19',
          # 'chrX'
          ]


def run_thread(no, chrom):
    for ch in chrom:
        dir_ = f'/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/embroy_without_common_graph/_{ch}'
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
                # loss = torch.mean((embeddings - graph_data.x) ** 2)
                loss = torch.var(embeddings)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                graph_data = graph_data.to('cpu')
                del graph_data

            print(f'{no}: Parent Epoch {epoch}, Average Loss: {total_loss / len(graph_list)}')
            if epoch % 1000 == 0 and epoch != 0:
                torch.save(model, f'{config_["parent_dir"]}/{ch}_embroy_no_features_{epoch}.pt')
                print(f'{no}: {ch} {epoch} saved')


thread_list = []
for i, items in enumerate(chrom_):
    thread_list.append(threading.Thread(target=run_thread, args=(i, [items])))

for thread in thread_list:
    thread.start()

for thread in thread_list:
    thread.join()
