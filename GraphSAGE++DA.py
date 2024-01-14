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
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super().__init__()
        self.num_layers = len(hidden_channels_list)
        self.convs_mean = torch.nn.ModuleList()
        self.convs_max = torch.nn.ModuleList()

        for i in range(self.num_layers):
            in_channels_layer = in_channels if i == 0 else 2 * hidden_channels_list[i-1]  # 注意这里的修改
            out_channels_layer = hidden_channels_list[i]
            self.convs_mean.append(SAGEConv(in_channels_layer, out_channels_layer, aggr='mean'))
            self.convs_max.append(SAGEConv(in_channels_layer, out_channels_layer, aggr='max'))

        # 输出层的维度需要考虑到拼接操作
        self.post_mp = nn.Linear(2 * hidden_channels_list[-1], out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        for conv in self.convs_max:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x_mean = self.convs_mean[i]((x, x_target), edge_index)
            x_max = self.convs_max[i]((x, x_target), edge_index)

            # 拼接mean和max层的结果
            x = torch.cat([x_mean, x_max], dim=1)

            if i != self.num_layers - 1:
                x = F.relu(x)

        # 在最后一层之后不需要ReLU
        x_final = self.post_mp(x)
        return F.log_softmax(x_final, dim=-1)

# 数据准备和模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0].to(device)

# 定义每层的维度，维度由dimension_calculate.py计算得出
hidden_dims = [73, 34, 21]
model = GraphSAGEPlusPlusDA(dataset.num_features, hidden_dims, dataset.num_classes).to(device)

train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[20, 10, 10], batch_size=64, shuffle=True)
test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[20, 10, 10], batch_size=64, shuffle=False)
# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练函数
def train():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:  # 正确处理从NeighborSampler返回的数据
        # 将邻居采样信息移动到正确的设备上
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        # 正确传递邻居采样数据给模型
        out = model(data.x[n_id].to(device), adjs)
        # 计算损失
        loss = criterion(out, data.y[n_id[:batch_size]].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 测试函数
def test():
    model.eval()
    correct = 0
    for batch_size, n_id, adjs in test_loader:  # 正确处理从NeighborSampler返回的数据
        adjs = [adj.to(device) for adj in adjs]  # 将邻居采样信息移动到正确的设备上

        out = model(data.x[n_id].to(device), adjs)  # 正确传递邻居采样数据给模型
        pred = out.argmax(dim=1)  # 获取概率最高的类别
        correct += int(pred.eq(data.y[n_id[:batch_size]].to(device)).sum().item())  # 计算正确预测的数量
    return correct / int(data.test_mask.sum())  # 返回测试准确率


# 训练和测试过程
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
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
