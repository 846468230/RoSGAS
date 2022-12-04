# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#
# import argparse
# import torch
# import torch.nn.functional as F
# from torch_geometric.datasets import Flickr
# from torch_geometric.data import GraphSAINTRandomWalkSampler
# from model.SocialData import SocialBotDataset
# from sklearn import preprocessing
# from torch_geometric.nn import GraphConv
# from torch_geometric.utils import degree
# import numpy as np
# import time
#
# general = True
# dataset_str = "botometer-feedback-2019"
# test_dataset = "varol-2017"
# #{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
#
# class Net(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(Net, self).__init__()
#         in_channels = dataset.num_node_features
#         out_channels = dataset.num_classes
#         self.conv1 = GraphConv(in_channels, hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.conv3 = GraphConv(hidden_channels, hidden_channels)
#         self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)
#
#     def set_aggr(self, aggr):
#         self.conv1.aggr = aggr
#         self.conv2.aggr = aggr
#         self.conv3.aggr = aggr
#
#     def forward(self, x0, edge_index, edge_weight=None):
#         x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
#         x1 = F.dropout(x1, p=0.2, training=self.training)
#         x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
#         x2 = F.dropout(x2, p=0.2, training=self.training)
#         x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
#         x3 = F.dropout(x3, p=0.2, training=self.training)
#         x = torch.cat([x1, x2, x3], dim=-1)
#         x = self.lin(x)
#         return x.log_softmax(dim=-1)
#
# def train():
#     model.train()
#     model.set_aggr('add' if args.use_normalization else 'mean')
#
#     total_loss = total_examples = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#
#         if args.use_normalization:
#             edge_weight = data.edge_norm * data.edge_weight
#             out = model(data.x, data.edge_index, edge_weight)
#             loss = F.nll_loss(out, data.y, reduction='none')
#             loss = (loss * data.node_norm)[dataset.train_index].sum()
#         else:
#             out = model(data.x, data.edge_index)
#             loss = F.nll_loss(out[dataset.train_index], data.y[dataset.train_index])
#
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_nodes
#         total_examples += data.num_nodes
#     return total_loss / total_examples
#
#
# @torch.no_grad()
# def test(general=True,dataset=None,test_dataset=None):
#     model.eval()
#     model.set_aggr('mean')
#     accs= []
#     for i, mask in enumerate([dataset.train_index, dataset.val_index, dataset.test_index]):
#         length = len(mask)
#         if i != 2:
#             pred = model(data.x, data.edge_index)[mask].max(1)[1]
#             acc = pred.eq(data.y[mask]).sum().item() / length
#         else:
#             mask = dataset.test_index
#             pred = model(dataset.test_data.x, dataset.test_data.edge_index)[dataset.test_index].max(1)[1]
#             acc = pred.eq(dataset.test_data.y[dataset.test_index]).sum().item() / length
#         accs.append(acc)
#     return accs
#
# K = 10
# test_accs = []
# times = []
# for item in range(K):
#     print(f"Start {item} fold")
#     min_max_scaler = preprocessing.MinMaxScaler()
#     dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform, K=item, General=general,
#                                dataset=dataset_str, test_dataset=test_dataset)
#     torch.manual_seed(12345)
#     # dataset = dataset.shuffle()
#     data = dataset[0]
#     start = time.time()
#     print(f'Number of training nodes: {len(dataset.train_index)}')
#     print(f'Number of test nodes: {len(dataset.test_index)}')
#     row, col = data.edge_index
#     data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use_normalization', action='store_true')
#     args = parser.parse_args()
#     loader = GraphSAINTRandomWalkSampler(data, batch_size=3000, walk_length=1,
#                                          num_steps=5, sample_coverage=100,
#                                          save_dir=dataset.processed_dir,
#                                          num_workers=0)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Net(hidden_channels=64).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.0005)
#     temp_accs = []
#     data.to(device)
#     dataset.test_data.to(device)
#     for epoch in range(1, 31):
#         loss = train()
#         accs = test(general=general,dataset=dataset,test_dataset=dataset.test_dataset)
#         temp_accs.append(accs[2])
#         print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
#             f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
#     test_accs.append(max(temp_accs))
#     end = time.time()
#     times.append(end - start)
#     print(f"GraphSAINT training time: {end - start}")
#     print(test_accs)
# print(f"GraphSAINT with 3 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
# print(f"GraphSAINT running time: {np.mean(times)}")

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree
import numpy as np
import time
general = False
dataset_str = "varol-2017"
test_dataset = "varol-2017"
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

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        features = x
        x = self.lin(x)
        return x.log_softmax(dim=-1),features.detach().cpu().numpy()

def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out,_ = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[dataset.train_index].sum()
        else:
            out,_ = model(data.x, data.edge_index)
            loss = F.nll_loss(out[dataset.train_index], data.y[dataset.train_index])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(general=True,dataset=None,test_dataset=None,epoch=0):
    model.eval()
    model.set_aggr('mean')
    accs= []
    for i, mask in enumerate([dataset.train_index, dataset.val_index, dataset.test_index]):
        length = len(mask)
        pred,agg_features = model(data.x, data.edge_index)
        pred = pred[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / length
        accs.append(acc)
        if epoch==30 and i==2:
            features = prepocessing_tsne(agg_features[mask],2)
            plot_figure(features,data.y[mask].cpu().numpy(),"GraphSAINT_"+dataset.cur_dataset)
            sys.exit(0)
        # if i == 2:
        #     hom_scores = cul_hom_score(agg_features[mask].detach().cpu().numpy(), data.y[mask].cpu().numpy())
    return accs#,hom_scores

K = 10
test_accs = []
home_scores=[]
times = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform, K=item, General=general,
                               dataset=dataset_str, test_dataset=test_dataset)
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    data = dataset[0]
    start = time.time()
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    row, col = data.edge_index
    data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_normalization', action='store_true')
    args = parser.parse_args()
    loader = GraphSAINTRandomWalkSampler(data, batch_size=3000, walk_length=1,
                                         num_steps=5, sample_coverage=100,
                                         save_dir=dataset.processed_dir,
                                         num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.0005)
    temp_accs = []
    data.to(device)
    # dataset.test_data.to(device)
    temp_hom_score = []
    for epoch in range(1, 31):
        loss = train()
        accs = test(general=general,dataset=dataset,test_dataset=None,epoch=epoch)#dataset.test_dataset)
        temp_accs.append(accs[2])
        # temp_hom_score.append(hom_score)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
            f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
    test_accs.append(max(temp_accs))
    home_scores.append(max(temp_hom_score))
    end = time.time()
    times.append(end - start)
    print(f"GraphSAINT training time: {end - start}")
    print(f"max hom score {max(temp_hom_score)}")
    print(test_accs)
print(f"GraphSAINT with 3 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
print(f"GraphSAINT running time: {np.mean(times)}")
print(f"GraphSAINT Mean Hom Score: {np.mean(home_scores)} , Var: {np.var(home_scores)}")