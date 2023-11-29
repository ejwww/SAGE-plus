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
import time
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import negative_sampling

class GraphSAGEPlusPlusDA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.convs_mean = torch.nn.ModuleList()
        self.convs_max = torch.nn.ModuleList()

        # 第一层
        self.convs_mean.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        self.convs_max.append(SAGEConv(in_channels, hidden_channels, aggr='max'))

        # 其他层
        for _ in range(num_layers - 1):
            self.convs_mean.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
            self.convs_max.append(SAGEConv(hidden_channels, hidden_channels, aggr='max'))

        # 输出层
        self.post_mp = nn.Linear(2 * hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        for conv in self.convs_max:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, adjs):
        x_mean = x
        x_max = x

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # 提取目标节点的特征

            # 对每层应用 mean 聚合
            x_mean = self.convs_mean[i]((x_mean, x_target), edge_index)

            # 对每层应用 max 聚合
            x_max = self.convs_max[i]((x_max, x_target), edge_index)

            if i == self.num_layers - 1:
                # 最后一层时合并 mean 和 max 的结果
                x = torch.cat([x_mean, x_max], dim=-1)
                x = self.post_mp(x)
                return F.log_softmax(x, dim=-1)
            else:
                x_mean = F.relu(x_mean)
                x_max = F.relu(x_max)


dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化
model = GraphSAGEPlusPlusDA(dataset.num_features, 32, dataset.num_classes, num_layers=3).to(device)

# 数据加载器
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[20, 10, 10], batch_size=64, shuffle=True)
test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[20, 10, 10], batch_size=64, shuffle=False)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs)
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

# 训练过程
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')


# 测试过程
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')


'''

data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

# 负采样
train_pos_edge_index = data.train_pos_edge_index
train_neg_edge_index = negative_sampling(
    edge_index=data.train_neg_edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=train_pos_edge_index.size(1)) 

def train():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # ... (与之前相同) ...

        edge_index, _ = adjs[0]
        pos_out = model(data.x[n_id].to(device), adjs)
        pos_out = pos_out[edge_index[0]] * pos_out[edge_index[1]]
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones(pos_out.size(0)).to(device))

        # 负采样
        neg_out = model(data.x[n_id].to(device), adjs)
        neg_out = neg_out[neg_edge_index[0]] * neg_out[neg_edge_index[1]]
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros(neg_out.size(0)).to(device))

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

'''
