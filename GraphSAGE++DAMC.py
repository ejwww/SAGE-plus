import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from torch_geometric.datasets import Planetoid


class GraphSAGEPlusPlusDAMC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs_mean = torch.nn.ModuleList()
        self.convs_max = torch.nn.ModuleList()

        # 添加一个初始的全连接层来调整特征维度
        self.initial_lin = nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers):
            conv_mean = pyg_nn.SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            conv_max = pyg_nn.SAGEConv(hidden_channels, hidden_channels, aggr='max')
            self.convs_mean.append(conv_mean)
            self.convs_max.append(conv_max)

        self.post_mp = nn.Linear(num_layers * 2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 通过初始全连接层调整输入特征的维度
        x = self.initial_lin(x)
        x = F.relu(x)  # 通常在全连接层后应用非线性激活函数

        all_layers = []

        for i in range(self.num_layers):
            x_mean = self.convs_mean[i](x, edge_index)
            x_max = self.convs_max[i](x, edge_index)

            x_mean = F.relu(x_mean)
            x_max = F.relu(x_max)

            # 不再进行全局平均，直接使用节点级别的特征
            all_layers.append(x_mean)
            all_layers.append(x_max)

        # 拼接所有层的输出
        x_final = torch.cat(all_layers, dim=1)
        x_final = self.post_mp(x_final)
        return F.log_softmax(x_final, dim=-1)


# 加载数据集
dataset = Planetoid(root='path/to/Planetoid', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GraphSAGEPlusPlusDAMC(dataset.num_features, 32, dataset.num_classes, num_layers=2).to(device)

# 训练过程
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    return correct / int(data.test_mask.sum())

# 训练和测试过程
for epoch in range(1, 201):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
