import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
import numpy as np
import time


general = True
# dataset = dataset.shuffle()
dataset_str = "botometer-feedback-2019"
test_dataset = "varol-2017"
#{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SGConv(
            dataset.num_features, dataset.num_classes, K=2, cached=True)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index)[dataset.train_index], data.y[dataset.train_index]).backward()
    optimizer.step()


def test(general=True,dataset=None,test_dataset=None):
    model.eval()
    accs =  []
    for i,mask in enumerate([dataset.train_index,dataset.val_index, dataset.test_index]):
        length = len(mask)
        if i != 2:
            pred = model(data.x, data.edge_index)[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / length
        else:
            mask = dataset.test_index
            pred = model(dataset.test_data.x, dataset.test_data.edge_index)[dataset.test_index].max(1)[1]
            acc = pred.eq(dataset.test_data.y[dataset.test_index]).sum().item() / length
        accs.append(acc)
    return accs

K = 10
test_accs = []
times = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform, K=item, General=general,
                               dataset=dataset_str, test_dataset=test_dataset)
    times = []
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    start = time.time()
    data = dataset[0]
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    device = torch.device('cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, )
    temp_accs = []
    for epoch in range(1, 31):
        train()
        train_acc, val_acc, test_acc = test(general=general,dataset=dataset,test_dataset=dataset.test_dataset)
        temp_accs.append(test_acc)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))
    test_accs.append(max(temp_accs))
    end = time.time()
    times.append(end - start)
    print(f"AMAR training time: {end - start}")
    print(test_accs)
print(f"SGC with 1 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
print(f"SGC running time: {np.mean(times)}")