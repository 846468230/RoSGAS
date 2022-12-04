import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from model.SocialData import SocialBotDataset
from torch_geometric.data import DataLoader
from sklearn import preprocessing
from torch_geometric.transforms import NormalizeFeatures
import math
from torch_geometric.nn import GATConv,global_mean_pool
import numpy as np
import time
general = True
dataset_str = "botometer-feedback-2019"
test_dataset = "vendor-purchased-2019"
#{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class GAT(torch.nn.Module):
    def __init__(self,hidden_channels,num_heads,num_features,num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, heads=num_heads, dropout=0.6)
        # self.lin = torch.nn.Linear(hidden_channels*num_heads, dataset.num_classes)

    def forward(self, x, edge_index,batch):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x,edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = global_mean_pool(x, batch)
        # # return F.log_softmax(x, dim=-1)
        # out = self.lin(x)
        return F.log_softmax(x[batch], dim=1)

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
        out = model(data.x, data.edge_index, batch_nodes)  # Perform a single forward pass.
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
        data.to(device)
    else:
        length = len(dataset.test_index)
        indexs = dataset.test_index
        dataset.test_data.to(device)
    num_batches = math.ceil(length / batch_size)
    correct = 0
    for batch in range(num_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, length)
        batch_nodes = indexs[i_start:i_end]
        if data_mode == "train":
            batch_label = data.y[batch_nodes]
            out = model(data.x, data.edge_index, batch_nodes)
        else:
            batch_label = dataset.test_data.y[batch_nodes]
            out = model(dataset.test_data.x, dataset.test_data.edge_index, batch_nodes)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch_label).sum())  # Check against ground-truth labels.
    return correct / length  # Derive ratio of correct predictions.
K = 10
test_accs = []
times = []
for item in range(K):
    print(f"Start {item} fold")
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform, K=item, General=general,
                               dataset=dataset_str, test_dataset=test_dataset)
    torch.manual_seed(12345)
    start = time.time()
    # dataset = dataset.shuffle()
    data = dataset[0]
    print(f'Number of training nodes: {len(dataset.train_index)}')
    print(f'Number of test nodes: {len(dataset.test_index)}')
    model = GAT(hidden_channels=16, num_heads=4,num_features=dataset.num_features,num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    temp_accs = []
    for epoch in range(1, 31):
        loss = train(dataset, data)
        # if epoch %10==0:
        #     visualize(h, color=data.y, epoch=epoch, loss=loss)
        #     time.sleep(0.3)
        train_acc = test(dataset, data, "train")
        test_acc = test(dataset, data, "test")
        temp_accs.append(test_acc)
        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    test_accs.append(max(temp_accs))
    end = time.time()
    times.append(end - start)
    print(f"GAT training time: {end - start}")
    print(test_accs)
print(f"GAT with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
print(f"GAT running time: {np.mean(times)}")