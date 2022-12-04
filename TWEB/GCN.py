# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# import torch
# from model.SocialData import SocialBotDataset
# from torch_geometric.data import DataLoader
# from torch.nn import Linear
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# import math
# from sklearn import preprocessing
# import numpy as np
# from torch_geometric.transforms import NormalizeFeatures
# from sklearn.preprocessing import scale
# from torch_geometric.nn import global_mean_pool,global_max_pool
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# import time
#
# general = True
# dataset_str = "botometer-feedback-2019"
# test_dataset = "vendor-purchased-2019"
# #{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
#
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels,num_features,num_classes):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.conv4 = GCNConv(hidden_channels, hidden_channels)
#         self.fc1 = torch.nn.Linear(hidden_channels, int(hidden_channels))
#         self.fc2 = torch.nn.Linear(int(hidden_channels), int(hidden_channels))
#         self.fc3 = torch.nn.Linear(int(hidden_channels), int(hidden_channels))
#         self.lin = Linear(int(hidden_channels),num_classes)
#
#     def forward(self, x, edge_index,batch_index):
#         # 1. Obtain node embeddings
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.lin(x))
#         # x = F.relu(self.conv3(x, edge_index))
#         # x = x.relu()
#         # x = F.relu(self.conv4(x, edge_index))
#         # x = F.dropout(F.relu(self.fc1(x)),p=0.5, training=self.training)
#         # x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
#         # x = F.dropout(F.relu(self.fc3(x)), p=0.5, training=self.training)
#         # 2. Readout layer
#         # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#
#         # 3. Apply a final classifier
#         # x = F.dropout(x, p=0.5, training=self.training)
#         out = x[batch_index]
#
#         return F.log_softmax(out, dim=1)
#
#
# batch_size = 32
#
# def train(dataset,data):
#     model.train()
#     num_batches = math.ceil(len(dataset.train_index) / batch_size)
#     loss = 0.0
#     epoch_time = 0
#     # mini-batch training
#     data.to(device)
#     for batch in range(num_batches):
#         i_start = batch * batch_size
#         i_end = min((batch + 1) * batch_size, len(dataset.train_index))
#         batch_nodes = dataset.train_index[i_start:i_end]
#         batch_label = data.y[batch_nodes]
#         out = model(data.x, data.edge_index, batch_nodes)  # Perform a single forward pass.
#         loss = criterion(out,batch_label)  # Compute the loss.
#         loss.backward()  # Derive gradients.
#         # print(loss.item())
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
#     return loss
#
# def test(dataset,data,data_mode):
#     model.eval()
#     if data_mode == "train":
#         length = len(dataset.train_index)
#         indexs = dataset.train_index
#         data.to(device)
#     else:
#         length = len(dataset.test_index)
#         indexs = dataset.test_index
#         dataset.test_data.to(device)
#     num_batches = math.ceil(length / batch_size)
#     correct = 0
#     for batch in range(num_batches):
#         i_start = batch * batch_size
#         i_end = min((batch + 1) * batch_size, length)
#         batch_nodes = indexs[i_start:i_end]
#         if data_mode == "train":
#             batch_label = data.y[batch_nodes]
#             out = model(data.x, data.edge_index, batch_nodes)
#         else:
#             batch_label = dataset.test_data.y[batch_nodes]
#             out = model(dataset.test_data.x, dataset.test_data.edge_index, batch_nodes)
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct += int((pred == batch_label).sum())  # Check against ground-truth labels.
#     return correct / length  # Derive ratio of correct predictions.
#
# K = 10
# test_accs = []
# for item in range(K):
#     print(f"Start {item} fold")
#     min_max_scaler = preprocessing.MinMaxScaler()
#     dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform,K=item,General=general,dataset=dataset_str,test_dataset=test_dataset)
#     torch.manual_seed(12345)
#     # dataset = dataset.shuffle()
#     start = time.time()
#     data = dataset[0]
#     print(f'Number of training nodes: {len(dataset.train_index)}')
#     print(f'Number of test nodes: {len(dataset.test_index)}')
#     model = GCN(hidden_channels=64, num_features=dataset.num_features,num_classes=dataset.num_classes).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
#     criterion = torch.nn.CrossEntropyLoss()
#     temp_accs = []
#     for epoch in range(1, 31):
#         loss = train(dataset,data)
#         # if epoch %10==0:
#         #     visualize(h, color=data.y, epoch=epoch, loss=loss)
#         #     time.sleep(0.3)
#         train_acc = test(dataset,data,"train")
#         test_acc = test(dataset,data,"test")
#         temp_accs.append(test_acc)
#         print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
#     test_accs.append(max(temp_accs))
#     print(test_accs)
#     end = time.time()
#     print(f"GCN training time: {end - start}")
# print(f"GCN with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from model.SocialData import SocialBotDataset
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
from sklearn import preprocessing
import numpy as np
from torch_geometric.transforms import NormalizeFeatures
from sklearn.preprocessing import scale
from torch_geometric.nn import global_mean_pool,global_max_pool
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import time
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

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, int(hidden_channels))
        self.fc2 = torch.nn.Linear(int(hidden_channels), int(hidden_channels))
        self.fc3 = torch.nn.Linear(int(hidden_channels), int(hidden_channels))
        self.lin = Linear(int(hidden_channels),num_classes)

    def forward(self, x, edge_index,batch_index):
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        features = x
        x = F.relu(self.lin(x))
        # x = F.relu(self.conv3(x, edge_index))
        # x = x.relu()
        # x = F.relu(self.conv4(x, edge_index))
        # x = F.dropout(F.relu(self.fc1(x)),p=0.5, training=self.training)
        # x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        # x = F.dropout(F.relu(self.fc3(x)), p=0.5, training=self.training)
        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        out = x[batch_index]

        return F.log_softmax(out, dim=1),out,features[batch_index].detach().cpu().numpy()


batch_size = 32

def train(dataset,data):
    model.train()
    num_batches = math.ceil(len(dataset.train_index) / batch_size)
    loss = 0.0
    epoch_time = 0
    # mini-batch training
    data.to(device)
    for batch in range(num_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(dataset.train_index))
        batch_nodes = dataset.train_index[i_start:i_end]
        batch_label = data.y[batch_nodes]
        out,_,_ = model(data.x, data.edge_index, batch_nodes)  # Perform a single forward pass.
        loss = criterion(out,batch_label)  # Compute the loss.
        loss.backward()  # Derive gradients.
        # print(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return loss

def test(dataset,data,data_mode):
    model.eval()
    if data_mode == "train":
        length = len(dataset.train_index)
        indexs = dataset.train_index
    else:
        length = len(dataset.test_index)
        indexs = dataset.test_index
    num_batches = math.ceil(length / batch_size)
    correct = 0
    data.to(device)
    features = None
    labels = None
    for batch in range(num_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, length)
        batch_nodes = indexs[i_start:i_end]
        batch_label = data.y[batch_nodes]
        out,agg_features,high_dim_features = model(data.x, data.edge_index, batch_nodes)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch_label).sum())  # Check against ground-truth labels.
        if features is None:
            features = agg_features.detach().cpu().numpy()
            high_features = high_dim_features
            labels = batch_label.cpu().numpy()
        else:
            high_features = np.concatenate((high_features, high_dim_features), axis=0)
            features = np.concatenate((features, agg_features.detach().cpu().numpy()), axis=0)
            labels = np.concatenate((labels, batch_label.cpu().numpy()))
    if data_mode == "test":
        return correct / length, cul_hom_score(features,labels),high_features,labels # Derive ratio of correct predictions.
    else:
        return correct / length, 0
K = 10
test_accs = []
hom_scores_real = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data",
                               pre_transform=min_max_scaler.fit_transform,K=item)  # transform=NormalizeFeatures())#pre_transform= min_max_scaler.fit_transform)
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    start = time.time()
    data = dataset[0]
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    model = GCN(hidden_channels=64, num_features=dataset.num_features,num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    temp_accs = []
    hom_scores= []
    for epoch in range(1, 31):
        loss = train(dataset,data)
        # if epoch %10==0:
        #     visualize(h, color=data.y, epoch=epoch, loss=loss)
        #     time.sleep(0.3)
        train_acc,_ = test(dataset,data,"train")
        test_acc,hom_score,high_features,labels= test(dataset,data,"test")
        temp_accs.append(test_acc)
        hom_scores.append(hom_score)
        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epoch == 30:
            x_tsne =  prepocessing_tsne(high_features,2)
            plot_figure(x_tsne,labels,"GCN_"+dataset.cur_dataset)
            sys.exit(0)
    test_accs.append(max(temp_accs))
    hom_scores_real.append(max(hom_scores))
    print(test_accs)
    end = time.time()
    print(f"GCN training time: {end - start}")
print(f"GCN with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
print(f"GCN with 2 layers Mean Hom Score: {np.mean(hom_scores_real)} , Var: {np.var(hom_scores_real)}")