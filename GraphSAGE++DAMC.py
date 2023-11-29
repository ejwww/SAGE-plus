import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import SAGEConv
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import PPI
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import negative_sampling
import time

class GraphSAGEPlusPlusDAMC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.convs_mean = torch.nn.ModuleList()
        self.convs_max = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs_mean.append(SAGEConv(in_channels if _ == 0 else hidden_channels, hidden_channels, aggr='mean'))
            self.convs_max.append(SAGEConv(in_channels if _ == 0 else hidden_channels, hidden_channels, aggr='max'))

        self.post_mp = nn.Linear(num_layers * 2 * hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        for conv in self.convs_max:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, adjs):
        all_layers = []

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x_mean = self.convs_mean[i]((x, x_target), edge_index)
            x_max = self.convs_max[i]((x, x_target), edge_index)

            all_layers.append(x_mean)
            all_layers.append(x_max)

            if i != self.num_layers - 1:
                x = F.relu(x)

        x_final = torch.cat(all_layers, dim=1)
        x_final = self.post_mp(x_final)
        return F.log_softmax(x_final, dim=-1)

# 数据准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0].to(device)

# 模型初始化
model = GraphSAGEPlusPlusDAMC(dataset.num_features, 32, dataset.num_classes, num_layers=3).to(device)
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[20, 10, 10], batch_size=64, shuffle=True)
test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[20, 10, 10], batch_size=64, shuffle=False)
# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
def test():
    model.eval()
    correct = 0
    for data in test_loader:
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        correct += int(pred[data.test_mask].eq(data.y[data.test_mask].to(device)).sum().item())
    return correct / int(data.test_mask.sum())

# 训练和测试过程
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')



'''
prediction
# 数据准备和边分割
data = train_test_split_edges(dataset[0])

data = data.to(device)

# 模型初始化
model = GraphSAGEPlusPlusDAC(dataset.num_features, 32, dataset.num_classes, num_layers=3).to(device)

# 定义损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x.to(device), data.train_pos_edge_index.to(device))

        # 正样本损失
        pos_loss = criterion(out[data.train_pos_edge_index], torch.ones(data.train_pos_edge_index.size(1)).to(device))

        # 负样本损失
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=data.train_pos_edge_index.size(1))
        neg_loss = criterion(out[neg_edge_index], torch.zeros(neg_edge_index.size(1)).to(device))

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x.to(device), data.test_pos_edge_index.to(device))
    # ... (链接预测的评估代码) ...

# 训练和测试过程
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
'''
