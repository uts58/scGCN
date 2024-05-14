import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from ucimlrepo import fetch_ucirepo

# Load data
parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features
y = parkinsons.data.targets
X = X.T.drop_duplicates().T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess data
scaler = StandardScaler()
features = scaler.fit_transform(X.values)
labels = y['status'].values

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

df = X.copy()

G = nx.Graph()

for idx, row in df.iterrows():
    print(features[idx])
    G.add_node(idx, x=features[idx], y=labels[idx])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

dist_matrix = euclidean_distances(features_scaled)

threshold = 1.0  # This is arbitrary; adjust based on domain knowledge
for i in range(len(dist_matrix)):
    for j in range(i + 1, len(dist_matrix)):
        if dist_matrix[i][j] < threshold:
            G.add_edge(i, j, weight=dist_matrix[i][j])

data = from_networkx(G)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data.to(device)

# Create masks for training and testing
num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Randomly assign indices to train or test sets
indices = np.random.permutation(num_nodes)
train_indices = indices[:int(0.7 * num_nodes)]
test_indices = indices[int(0.7 * num_nodes):]

train_mask[train_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask

# DataLoader (We use the entire data as a single batch in this case)
loader = DataLoader([data], batch_size=1, shuffle=False)


# Define the GCN model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# Initialize model and optimizer
model = GCN(num_features=data.num_features, hidden_channels=128, num_classes=y['status'].nunique())
model = model.to(device)  # Move model to GPU

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# Training function
def train():
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return loss


# Test function
def test(mask):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += pred[mask].eq(data.y[mask]).sum().item()
    return correct / mask.sum().item()


# Test function to calculate accuracy and other metrics
def evaluate(mask):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        y_true.extend(data.y[mask].tolist())
        y_pred.extend(pred[mask].tolist())
    acc = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    return y_true, y_pred, acc


# Function to plot the learning curve from the metrics dictionary
def plot_learning_curve(metrics):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(metrics["epochs"], metrics["train_losses"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(metrics["epochs"], metrics["train_accs"], label='Train Accuracy', color='blue')
    ax2.plot(metrics["epochs"], metrics["test_accs"], label='Test Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# Train and evaluate the model, collect data for plots
metrics = {
    "train_losses": [],
    "train_accs": [],
    "test_accs": [],
    "epochs": [],
    "confusion_matrices": [],
    "train_f1_scores": [],
    "train_precisions": [],
    "train_recalls": []
}

for epoch in range(5000):
    loss = train()
    if epoch % 10 == 0:
        train_y_true, train_y_pred, train_acc = evaluate(data.train_mask)
        test_y_true, test_y_pred, test_acc = evaluate(data.test_mask)

        # Collect and store metrics
        metrics["train_losses"].append(loss.item())
        metrics["train_accs"].append(train_acc)
        metrics["test_accs"].append(test_acc)
        metrics["epochs"].append(epoch)
        metrics["confusion_matrices"].append(confusion_matrix(test_y_true, test_y_pred))
        metrics["train_f1_scores"].append(f1_score(train_y_true, train_y_pred, average="weighted"))
        metrics["train_precisions"].append(precision_score(train_y_true, train_y_pred, average="weighted"))
        metrics["train_recalls"].append(recall_score(train_y_true, train_y_pred, average="weighted"))

        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        # print(f'F1-Score (Train): {metrics["train_f1_scores"][-1]:.4f}')
        # print(f'Precision (Train): {metrics["train_precisions"][-1]:.4f}')
        # print(f'Recall (Train): {metrics["train_recalls"][-1]:.4f}')
        # print(f'Confusion Matrix (Train):\n{confusion_matrix(train_y_true, train_y_pred)}')

# Plotting the learning curve
plot_learning_curve(metrics)
