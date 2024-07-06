import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

class GraphSAGEPlusPlusDAMC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super().__init__()
        self.num_layers = len(hidden_channels_list)

        self.convs_mean = torch.nn.ModuleList()
        self.convs_max = torch.nn.ModuleList()

        self.initial_lin = nn.Linear(in_channels, hidden_channels_list[0])

        for i in range(self.num_layers):
            in_channels_layer = hidden_channels_list[i - 1] if i != 0 else hidden_channels_list[0]
            out_channels_layer = hidden_channels_list[i]
            self.convs_mean.append(SAGEConv(in_channels_layer, out_channels_layer, aggr='mean'))
            self.convs_max.append(SAGEConv(in_channels_layer, out_channels_layer, aggr='max'))

        self.post_mp = nn.Linear(sum(hidden_channels_list) * 2, out_channels)
        self.adjust_layer = nn.Linear(hidden_channels_list[0], hidden_channels_list[0])

    def forward(self, x, adjs):
        x = F.relu(self.initial_lin(x))

        all_layers = []
        max_num_nodes = max([size[1] for _, _, size in adjs])

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x_mean = F.relu(self.convs_mean[i](x, edge_index))
            x_max = F.relu(self.convs_max[i](x, edge_index))

            x_mean = self.adjust_layer(x_mean)
            x_max = self.adjust_layer(x_max)

            # 使用零填充确保所有层的输出节点数一致
            if x_mean.size(0) < max_num_nodes:
                pad_size = max_num_nodes - x_mean.size(0)
                x_mean = F.pad(x_mean, (0, 0, 0, pad_size))
                x_max = F.pad(x_max, (0, 0, 0, pad_size))

            all_layers.append(x_mean)
            all_layers.append(x_max)

        x_final = torch.cat(all_layers, dim=1)
        x_final = self.post_mp(x_final)
        return F.log_softmax(x_final[:max_num_nodes], dim=-1)

dataset = Planetoid(root='path/to/Planetoid', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
hidden_dims = [32, 32]
model = GraphSAGEPlusPlusDAMC(dataset.num_features, hidden_dims, dataset.num_classes).to(device)
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[20, 10], batch_size=64, shuffle=True)
test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[20, 10], batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs)
        loss = criterion(out[:batch_size], data.y[n_id[:batch_size]].to(device))  # 裁剪输出
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test():
    model.eval()
    correct = 0
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        out = model(data.x[n_id].to(device), adjs)
        pred = out[:batch_size].argmax(dim=1)  # 裁剪输出
        correct += int(pred.eq(data.y[n_id[:batch_size]].to(device)).sum().item())
    return correct / int(data.test_mask.sum())

for epoch in range(1, 401):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
