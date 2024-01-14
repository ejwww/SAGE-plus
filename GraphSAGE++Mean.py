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

class GraphSAGEPlusPlusMean(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super().__init__()
        self.num_layers = len(hidden_channels_list)
        self.convs_mean = torch.nn.ModuleList()

        for i in range(self.num_layers):
            in_channels_layer = in_channels if i == 0 else hidden_channels_list[i - 1]
            self.convs_mean.append(SAGEConv(in_channels_layer, hidden_channels_list[i], aggr='mean'))

        self.post_mp = nn.Linear(sum(hidden_channels_list), out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, adjs):
        all_layers_output = []

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs_mean[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)

            all_layers_output.append(x)

        x_final = torch.cat(all_layers_output, dim=1)
        x_final = self.post_mp(x_final)
        return F.log_softmax(x_final, dim=-1)

# 数据准备和模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0].to(device)

hidden_dims = [73, 34, 21]  # 定义每层的维度
model = GraphSAGEPlusPlusMean(dataset.num_features, hidden_dims, dataset.num_classes).to(device)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 数据加载器
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[20, 10, 10], batch_size=64, shuffle=True)
test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[20, 10, 10], batch_size=64, shuffle=False)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # 注意：adjs 是一个列表，其中每个元素是一个包含三个部分的元组
        adjs = [adj.to(device) for adj in adjs]  # 将每个元组的每个部分移动到设备上
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs)  # 注意：这里使用了 adjs
        loss = criterion(out, data.y[n_id[:batch_size]].to(device))
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
        pred = out.argmax(dim=1)
        correct += int(pred.eq(data.y[n_id[:batch_size]].to(device)).sum().item())
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

# initial
model = GraphSAGEPlusPlusDAC(dataset.num_features, 32, dataset.num_classes, num_layers=3).to(device)

# optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# train
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x.to(device), data.train_pos_edge_index.to(device))

        # positive
        pos_loss = criterion(out[data.train_pos_edge_index], torch.ones(data.train_pos_edge_index.size(1)).to(device))

        # negtive
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=data.train_pos_edge_index.size(1))
        neg_loss = criterion(out[neg_edge_index], torch.zeros(neg_edge_index.size(1)).to(device))

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# test
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x.to(device), data.test_pos_edge_index.to(device))
    # ... (链接预测的评估代码) ...

# train and test
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
'''
