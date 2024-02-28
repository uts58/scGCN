import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_layers):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, embedding_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class ModelDeep(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_layers):
        super(ModelDeep, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, embedding_size)

        self.conv2_bn = nn.BatchNorm1d(embedding_size, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)
        self.fc_block1 = nn.Linear(embedding_size, 10)
        self.fc_block2 = nn.Linear(10, 5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        x = F.leaky_relu(x)
        x = self.conv2_bn(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(self.fc_block1(x))
        x = self.fc_block2(x)
        return x
