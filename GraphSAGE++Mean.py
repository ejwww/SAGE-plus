import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_geometric.datasets import Planetoid

class GraphSAGEPlusPlusDAC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super().__init__()
        self.num_layers = len(hidden_channels_list)
        self.convs_mean = torch.nn.ModuleList()

        # 第一层的初始化
        self.convs_mean.append(SAGEConv(in_channels, hidden_channels_list[0], aggr='mean'))

        # 其余层的初始化
        for i in range(1, self.num_layers):
            self.convs_mean.append(SAGEConv(hidden_channels_list[i-1], hidden_channels_list[i], aggr='mean'))

        # 输出层
        self.post_mp = nn.Linear(hidden_channels_list[-1], out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs_mean[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        x_final = self.post_mp(x)
        return F.log_softmax(x_final, dim=-1)

# 数据准备和模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0].to(device)

# 定义每层的维度，例如 [32, 64, 128]
hidden_dims = [32, 64, 128]
model = GraphSAGEPlusPlusDAC(dataset.num_features, hidden_dims, dataset.num_classes).to(device)

# ...[其余代码保持不变]...


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

