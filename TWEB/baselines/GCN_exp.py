import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, num_classes)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, int(hidden_channels))
        self.fc2 = torch.nn.Linear(int(hidden_channels), int(hidden_channels))
        self.fc3 = torch.nn.Linear(int(hidden_channels), int(hidden_channels))
        self.lin = Linear(int(hidden_channels),num_classes)

    def forward(self, x, edge_index,batch_index):
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        # # x = F.relu(self.lin(x))
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

        return F.log_softmax(out, dim=1)


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
    else:
        length = len(dataset.test_index)
        indexs = dataset.test_index
    num_batches = math.ceil(length / batch_size)
    correct = 0
    data.to(device)
    for batch in range(num_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, length)
        batch_nodes = indexs[i_start:i_end]
        batch_label = data.y[batch_nodes]
        out = model(data.x, data.edge_index, batch_nodes)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch_label).sum())  # Check against ground-truth labels.
    return correct / length  # Derive ratio of correct predictions.

def test_sample(dataset,data,records,record):
    model.eval()
    data.to(device)
    nodes = dataset.samlple_index
    label = dataset.samlple_label.to(device)
    out = model(data.x, data.edge_index, nodes)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = list(pred == label)  # Check against ground-truth labels.
    correct = [ int(item.item()) for item in correct]
    if record:
        records.append(correct)
    print(correct)
    return correct  # Derive ratio of correct predictions
min_max_scaler = preprocessing.MinMaxScaler()
dataset = SocialBotDataset(root="./data",
                               pre_transform=min_max_scaler.fit_transform,K=0)  # transform=NormalizeFeatures())#pre_transform= min_max_scaler.fit_transform)
torch.manual_seed(12345)
# dataset = dataset.shuffle()
data = dataset[0]
print(f'Number of training nodes: {len(dataset.train_index)}')
print(f'Number of test nodes: {len(dataset.test_index)}')
K = 100
test_accs = []
records= []
for item in range(K):
    print(f"Start {item} fold")
    model = GCN(hidden_channels=64, num_features=dataset.num_features,num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    temp_accs = []
    for epoch in range(1, 6):
        loss = train(dataset,data)
        # if epoch %10==0:
        #     visualize(h, color=data.y, epoch=epoch, loss=loss)
        #     time.sleep(0.3)
        # train_acc = test(dataset,data,"train")
        # test_acc = test(dataset,data,"test")
        # temp_accs.append(test_acc)
        # print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epoch == 5:
            test_sample(dataset,data,records=records,record = True)
        else:
            test_sample(dataset,data,records=records,record = False)

    # test_accs.append(max(temp_accs))
    # print(test_accs)
# print(f"GCN with 2 layers Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
import pickle
import pprint
pprint.pprint(records)
with open('1layers.pickle', 'wb') as f:
    pickle.dump(records, f)
records = np.array(records)