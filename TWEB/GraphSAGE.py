# import os.path as osp
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_cluster import random_walk
# from sklearn.linear_model import LogisticRegression
# from model.SocialData import SocialBotDataset
# from sklearn import preprocessing
# import torch_geometric.transforms as T
# from torch_geometric.nn import SAGEConv
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import NeighborSampler as RawNeighborSampler
# import numpy as np
# EPS = 1e-15
# general = False
# dataset_str = "cresci-2015"
# test_dataset = "vendor-purchased-2019"
# #{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
#
# from sklearn.cluster import  KMeans
# from sklearn.metrics import homogeneity_score
#
# def cul_hom_score(aggregated_features,labels):
#     kmeans = KMeans(n_clusters=2)
#     kmeans.fit(aggregated_features)
#     pre_labels = kmeans.predict(aggregated_features)
#     hom_score = homogeneity_score(labels, pre_labels)
#     print(f"homogeneity score is: {homogeneity_score(labels, pre_labels)}")
#     return hom_score
#
# class NeighborSampler(RawNeighborSampler):
#     def sample(self, batch):
#         batch = torch.tensor(batch)
#         row, col, _ = self.adj_t.coo()
#
#         # For each node in `batch`, we sample a direct neighbor (as positive
#         # example) and a random node (as negative example):
#         pos_batch = random_walk(row, col, batch, walk_length=1,
#                                 coalesced=False)[:, 1]
#
#         neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
#                                   dtype=torch.long)
#
#         batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
#         return super(NeighborSampler, self).sample(batch)
#
#
# class SAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers):
#         super(SAGE, self).__init__()
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         for i in range(num_layers):
#             in_channels = in_channels if i == 0 else hidden_channels
#             self.convs.append(SAGEConv(in_channels, hidden_channels))
#
#     def forward(self, x, adjs):
#         for i, (edge_index, _, size) in enumerate(adjs):
#             x_target = x[:size[1]]  # Target nodes are always placed first.
#             x = self.convs[i]((x, x_target), edge_index)
#             if i != self.num_layers - 1:
#                 x = x.relu()
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x
#
#     def full_forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if i != self.num_layers - 1:
#                 x = x.relu()
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x
#
# def train(dataset,data):
#     model.train()
#
#     total_loss = 0
#     for batch_size, n_id, adjs in train_loader:
#         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
#         adjs = [adj.to(device) for adj in adjs]
#         optimizer.zero_grad()
#
#         out = model(x[n_id], adjs)
#         out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
#
#         pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
#         neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
#         loss = -pos_loss - neg_loss
#         loss.backward()
#         optimizer.step()
#
#         total_loss += float(loss) * out.size(0)
#
#     return total_loss / data.num_nodes
#
#
# @torch.no_grad()
# def test(dataset,data):
#     model.eval()
#     out = model.full_forward(x, edge_index).cpu()
#
#     clf = LogisticRegression()
#     clf.fit(out[dataset.train_index], data.y[dataset.train_index])
#
#     val_acc = clf.score(out[dataset.val_index], data.y[dataset.val_index])
#     # dataset.test_data.to(device)
#     # out = model.full_forward(dataset.test_data.x,dataset.test_data.edge_index).cpu()
#     homescore = cul_hom_score(out[dataset.test_index].detach().cpu().numpy(),data.y.cpu()[dataset.test_index].numpy())
#     test_acc = clf.score(out[dataset.test_index], data.y.cpu()[dataset.test_index])
#
#     return val_acc, test_acc,homescore
#
# K = 10
# test_accs = []
# home_scores = []
# for item in range(K):
#     print(f"Start {item} fold")
#     min_max_scaler = preprocessing.MinMaxScaler()
#     dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform,K=item,General=general,dataset=dataset_str,test_dataset=test_dataset)
#     torch.manual_seed(12345)
#     # dataset = dataset.shuffle()
#     data = dataset[0]
#     print(f'Number of training nodes: {len(dataset.train_index)}')
#     print(f'Number of test nodes: {len(dataset.test_index)}')
#     train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
#                                    shuffle=True, node_idx=torch.cat((dataset.train_index, dataset.val_index)),
#                                    num_nodes=data.num_nodes)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = SAGE(data.num_node_features, hidden_channels=16, num_layers=2)
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     x, edge_index = data.x.to(device), data.edge_index.to(device)
#     temp_accs = []
#     temp_homescores = []
#     for epoch in range(1, 31):
#         loss = train(dataset,data)
#         val_acc, test_acc,homescore = test(dataset,data)
#         temp_accs.append(test_acc)
#         temp_homescores.append(homescore)
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#           f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
#     test_accs.append(max(temp_accs))
#     home_scores.append(max(temp_homescores))
#     print(test_accs)
# print(f"GraphSAGE with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
# print(f"GraphSAGE with 2 layers Mean Hom Score: {np.mean(home_scores)} , Var: {np.var(home_scores)}")

import os.path as osp
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler as RawNeighborSampler
import numpy as np
EPS = 1e-15
general = False
dataset_str = "varol-2017"
test_dataset = "vendor-purchased-2019"
#{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
def prepocessing_tsne(data, n):
    starttime_tsne = time.time()
    dataset = TSNE(n_components=n, random_state=33).fit_transform(data)
    endtime_tsne = time.time()
    print('cost time by tsne:', endtime_tsne - starttime_tsne)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tsne = scaler.fit_transform(dataset)
    return X_tsne

def plot_figure(digits_tsne,target,name='1'):
    # sns.set_style("darkgrid")  # 设立风格
    # plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    human_index = target==0
    bot_index = target==1
    plt.scatter(digits_tsne[human_index, 0], digits_tsne[human_index, 1], c=[(34/255, 255/255, 4/255),],edgecolors=[(102/255, 159/255, 36/255),])#, alpha=0.6,)
    plt.scatter(digits_tsne[bot_index, 0], digits_tsne[bot_index, 1], c=[(252/255, 0/255, 5/255),],edgecolors=[(245/255, 99/255, 104/255),])#, alpha=0.6, )
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    #plt.title("digits t-SNE", fontsize=18)
    #cbar = plt.colorbar(ticks=range(10))
    #cbar.set_label(label='digit value', fontsize=18)
    plt.legend(("Human", "Bot"), loc="upper right",fontsize=36,frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{name}.pdf', dpi=600)

from sklearn.cluster import  KMeans
from sklearn.metrics import homogeneity_score

def cul_hom_score(aggregated_features,labels):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(aggregated_features)
    pre_labels = kmeans.predict(aggregated_features)
    hom_score = homogeneity_score(labels, pre_labels)
    print(f"homogeneity score is: {homogeneity_score(labels, pre_labels)}")
    return hom_score

class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

def train(dataset,data):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test(dataset,data,epoch):
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[dataset.train_index], data.y[dataset.train_index])

    val_acc = clf.score(out[dataset.val_index], data.y[dataset.val_index])
    # dataset.test_data.to(device)
    # out = model.full_forward(dataset.test_data.x,dataset.test_data.edge_index).cpu()
    if epoch==30:
        features = prepocessing_tsne(out[dataset.test_index].detach().cpu().numpy(),2)
        plot_figure(features,data.y.cpu()[dataset.test_index].numpy(),"GraphSAGE_"+dataset.cur_dataset)
        sys.exit(0)
    homescore = cul_hom_score(out[dataset.test_index].detach().cpu().numpy(),data.y.cpu()[dataset.test_index].numpy())
    test_acc = clf.score(out[dataset.test_index], data.y.cpu()[dataset.test_index])

    return val_acc, test_acc,homescore

K = 10
test_accs = []
home_scores = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform,K=item,General=general,dataset=dataset_str,test_dataset=test_dataset)
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    data = dataset[0]
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                                   shuffle=True, node_idx=torch.cat((dataset.train_index, dataset.val_index)),
                                   num_nodes=data.num_nodes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(data.num_node_features, hidden_channels=16, num_layers=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    temp_accs = []
    temp_homescores = []
    for epoch in range(1, 31):
        loss = train(dataset,data)
        val_acc, test_acc,homescore = test(dataset,data,epoch)
        temp_accs.append(test_acc)
        temp_homescores.append(homescore)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    test_accs.append(max(temp_accs))
    home_scores.append(max(temp_homescores))
    print(test_accs)
print(f"GraphSAGE with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
print(f"GraphSAGE with 2 layers Mean Hom Score: {np.mean(home_scores)} , Var: {np.var(home_scores)}")