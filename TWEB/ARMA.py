# import os.path as osp
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# import torch
# import torch.nn.functional as F
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from torch_geometric.nn import ARMAConv
# from model.SocialData import SocialBotDataset
# from sklearn import preprocessing
# import numpy as np
# min_max_scaler = preprocessing.MinMaxScaler()
# torch.manual_seed(12345)
# import time
# # dataset = dataset.shuffle()
# dataset_str = "botometer-feedback-2019"
# test_dataset = "varol-2017"
# #{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
#
#
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.conv1 = ARMAConv(dataset.num_features, 16, num_stacks=2,
#                               num_layers=1, shared_weights=True, dropout=0.25)
#
#         self.conv2 = ARMAConv(16, dataset.num_classes, num_stacks=2,
#                               num_layers=1, shared_weights=True, dropout=0.25,
#                               act=lambda x: x)
#
#     def forward(self, x, edge_index):
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
#
# def train():
#     model.train()
#     optimizer.zero_grad()
#     F.nll_loss(model(data.x, data.edge_index)[dataset.train_index], data.y[dataset.train_index]).backward()
#     optimizer.step()
#
#
# def test(general=True,dataset=None,test_dataset=None):
#     model.eval()
#     accs =  []
#     for i,mask in enumerate([dataset.train_index,dataset.val_index, dataset.test_index]):
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
# general = True
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
#     start = time.time()
#     data = dataset[0]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataset.test_data.to(device)
#     print(f'Number of training nodes: {len(dataset.train_index)}')
#     print(f'Number of test nodes: {len(dataset.test_index)}')
#
#     model, data = Net().to(device), data.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#     temp_accs = []
#     for epoch in range(1, 31):
#         train()
#         train_acc, val_acc, test_acc = test(general=general,dataset=dataset,test_dataset=dataset.test_dataset)
#         temp_accs.append(test_acc)
#         log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#         print(log.format(epoch, train_acc, val_acc, test_acc))
#     test_accs.append(max(temp_accs))
#     end = time.time()
#     times.append(end-start)
#     print(f"AMAR training time: {end - start}")
#     print(test_accs)
# print(f"ARMA with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
# print(f"AMRA running time: {np.mean(times)}")

import os.path as osp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()
# dataset = SocialBotDataset(root="./data",pre_transform=min_max_scaler.fit_transform,transer_y=True)#transform=NormalizeFeatures())#pre_transform= min_max_scaler.fit_transform)
torch.manual_seed(12345)
import time
# dataset = dataset.shuffle()
# data = dataset[0]
# print(f'Number of training nodes: {len(dataset.train_index)}')
# print(f'Number of test nodes: {len(dataset.test_index)}')
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
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


def plot_figure(digits_tsne, target, name='1'):
    # sns.set_style("darkgrid")  # 设立风格
    # plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    human_index = target == 0
    bot_index = target == 1
    plt.scatter(digits_tsne[human_index, 0], digits_tsne[human_index, 1], c=[(34 / 255, 255 / 255, 4 / 255), ],
                edgecolors=[(102 / 255, 159 / 255, 36 / 255), ])  # , alpha=0.6,)
    plt.scatter(digits_tsne[bot_index, 0], digits_tsne[bot_index, 1], c=[(252 / 255, 0 / 255, 5 / 255), ],
                edgecolors=[(245 / 255, 99 / 255, 104 / 255), ])  # , alpha=0.6, )
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    # plt.title("digits t-SNE", fontsize=18)
    # cbar = plt.colorbar(ticks=range(10))
    # cbar.set_label(label='digit value', fontsize=18)
    plt.legend(("Human", "Bot"), loc="upper right", fontsize=36, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{name}.pdf', dpi=600)


from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score


def cul_hom_score(aggregated_features, labels):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(aggregated_features)
    pre_labels = kmeans.predict(aggregated_features)
    hom_score = homogeneity_score(labels, pre_labels)
    print(f"homogeneity score is: {homogeneity_score(labels, pre_labels)}")
    return hom_score


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = ARMAConv(dataset.num_features, 16, num_stacks=2,
                              num_layers=1, shared_weights=True, dropout=0.25)

        self.conv2 = ARMAConv(16, dataset.num_classes, num_stacks=2,
                              num_layers=1, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        agg_feature = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), agg_feature.detach().cpu().numpy()


def train():
    model.train()
    optimizer.zero_grad()

    F.nll_loss(model(data.x, data.edge_index)[0][dataset.train_index], data.y[dataset.train_index]).backward()
    optimizer.step()


def test():
    model.eval()
    hom_scores = None
    accs = []
    for i, mask in enumerate([dataset.train_index, dataset.val_index, dataset.test_index]):
        length = len(mask)
        pred, agg_features = model(data.x, data.edge_index)
        pred = pred[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / length
        accs.append(acc)
        del pred
        # if i == 2:
        #     hom_scores=cul_hom_score(agg_features[mask].detach().cpu().numpy(),data.y[mask].cpu().numpy())
    return *accs, data.y[mask].cpu().numpy(), agg_features[mask.cpu().numpy()]


K = 10
test_accs = []
times = []
hom_scores_real = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data",
                               pre_transform=min_max_scaler.fit_transform, transer_y=True,
                               K=item)  # transform=NormalizeFeatures())#pre_transform= min_max_scaler.fit_transform)
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    start = time.time()
    data = dataset[0]
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    temp_accs = []
    hom_scores = []
    for epoch in range(1, 31):
        train()
        train_acc, val_acc, test_acc, label, features = test()
        temp_accs.append(test_acc)
        # hom_scores.append(hom_score)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))
        if epoch == 30:
            x_tsne = prepocessing_tsne(features, 2)
            plot_figure(x_tsne, label, "ARMA_" + dataset.cur_dataset)
            import sys
            sys.exit(0)
    test_accs.append(max(temp_accs))
    end = time.time()
    times.append(end - start)
    hom_scores_real.append(max(hom_scores))
    print(f"AMAR training time: {end - start}")
    print(test_accs)
print(f"ARMA with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
print(f"AMRA running time: {np.mean(times)}")
print(f"GCN with 2 layers Mean Hom Score: {np.mean(hom_scores_real)} , Var: {np.var(hom_scores_real)}")
