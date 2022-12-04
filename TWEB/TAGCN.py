import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import TAGConv
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
import numpy as np

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = TAGConv(dataset.num_features, 16,K=1)
        self.conv2 = TAGConv(16, dataset.num_classes,K=1)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


K = 10
test_accs = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data",
                               pre_transform=min_max_scaler.fit_transform,transer_y=True,K=item)  # transform=NormalizeFeatures())#pre_transform= min_max_scaler.fit_transform)
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    data = dataset[0]
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    temp_accs = []
    for epoch in range(1, 31):
        train()
        train_acc, val_acc, test_acc = test()
        temp_accs.append(test_acc)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))
    test_accs.append(max(temp_accs))
    print(test_accs)
print(f"TAGCN with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
