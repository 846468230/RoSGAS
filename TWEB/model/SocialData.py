import random
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric
import os
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import remove_self_loops
from sklearn.model_selection import KFold
np.random.seed(12345)

class SocialBotDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,validation_ratio=0.5,test_ratio=0.9,dataset="vendor-purchased-2019",transer_y=False,KFold=10,K=0,General=False,test_dataset="botometer-feedback-2019"):
        self.datasets = {"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"gilani-2017":3,"vendor-purchased-2019":4,"varol-2017":5}
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.cur_dataset = dataset
        super(SocialBotDataset, self).__init__(root, transform, pre_transform)
        if not General:
            self.data, self.slices,self.G = torch.load(self.processed_paths[self.datasets[self.cur_dataset]])
            labels = self.data.y
            self.KFold = KFold
            self.K = K
            assert self.K<self.KFold
            self.transer_y = transer_y
            self.split_data(labels)
        else:
            self.data, self.slices, self.G = torch.load(self.processed_paths[self.datasets[self.cur_dataset]])
            self.test_dataset = test_dataset
            self.test_data,_,_ = torch.load(self.processed_paths[self.datasets[self.test_dataset]])
            labels = self.data.y
            test_labels = self.test_data.y
            self.KFold = KFold
            self.K = K
            assert self.K < self.KFold
            self.transer_y = transer_y
            self.general_split_data(labels,test_labels)

    @property
    def raw_file_names(self):
        return ["cresci-2015_sub.pickle","botometer-feedback-2019_sub.pickle","cresci-rtbust-2019_sub.pickle","gilani-2017_sub.pickle","vendor-purchased-2019_sub.pickle","varol-2017_sub.pickle"]

    @property
    def processed_file_names(self):
        return ['cresci-2015.pt','botometer-feedback-2019.pt','cresci-rtbust-2019.pt','gilani-2017.pt','vendor-purchased-2019.pt','varol-2017.pt']

    def download(self):
        pass

    def process(self):
        filepath = os.path.join("data","raw",self.raw_file_names[self.datasets[self.cur_dataset]])
        data_list = []
        with open(filepath, 'rb') as f:
            node_features, G, labels = pickle.load(f)
        node_features = np.array(node_features)
        if self.pre_transform is not None:
            features = node_features[:,0:300]
        else:
            features = node_features
        edge_index = np.array(G.edges).T
        edge_index = remove_self_loops(edge_index)
        data_list.append(Data(x=torch.tensor(features.astype(np.float32)), edge_index=torch.tensor(edge_index,dtype=torch.long), y=torch.tensor(labels,dtype=torch.long)))
        data, slices = self.collate(data_list)
        torch.save((data, slices,G), self.processed_paths[self.datasets[self.cur_dataset]])

    def split_data(self,labels):
        kf = KFold(n_splits=self.KFold, shuffle=True,random_state=12345)
        test_index,train_index=list(kf.split(list(range(len(labels)))))[self.K]
        train_index, test_index, y_train, y_test = train_index,test_index,labels[train_index],labels[test_index]
        train_index,val_index,y_train,y_validation = train_test_split(train_index,y_train,stratify=y_train,test_size=self.validation_ratio,random_state=12345,shuffle=True)
        sample_index = np.random.choice(test_index,50,replace=False)
        self.samlple_index = torch.tensor(sample_index)
        self.samlple_label = labels[self.samlple_index]
        if self.transer_y:
            data = self[0]
            data.y = torch.cat((labels,torch.tensor([0]*(data.x.size(0)-len(labels)),dtype=torch.long)))
            data.train_mask = index_to_mask(train_index, size=data.x.size(0))
            data.val_mask=index_to_mask(val_index, size=data.x.size(0))
            data.test_mask=index_to_mask(test_index, size=data.x.size(0))
            if torch_geometric.__version__=='1.7.0':
                self.__data_list__[0] = data
            else:
                self._data_list[0] = data
        self.train_index = torch.tensor(train_index)
        self.val_index = torch.tensor(val_index)
        self.test_index = torch.tensor(test_index)
        print(f'Train and test on {self.cur_dataset}.')
        print(f'Number of edges: {self[0].edge_index.size(1)}')
        print(f'Number of nodes: {self[0].x.size(0)}')
        print(f'Number of labeled nodes: {len(labels)}')
        print(f'Number of social bots: {sum(labels)}')
        print(f'Number of training nodes: {len(self.train_index)}')
        print(f'Number of test nodes: {len(self.test_index)}')

    def general_split_data(self,labels,test_labels):
        kf = KFold(n_splits=self.KFold, shuffle=True,random_state=12345)
        _,train_index=list(kf.split(list(range(len(labels)))))[self.K]
        test_index,_ =list(kf.split(list(range(len(test_labels)))))[self.K]
        train_index, test_index, y_train, y_test = train_index,test_index,labels[train_index],test_labels[test_index]
        train_index,val_index,y_train,y_validation = train_test_split(train_index,y_train,stratify=y_train,test_size=self.validation_ratio,random_state=12345,shuffle=True)
        sample_index = np.random.choice(test_index,50,replace=False)
        if self.transer_y:
            data = self[0]
            data.y = torch.cat((labels,torch.tensor([0]*(data.x.size(0)-len(labels)),dtype=torch.long)))
            data.train_mask = index_to_mask(train_index, size=data.x.size(0))
            data.val_mask=index_to_mask(val_index, size=data.x.size(0))
            data.test_mask=index_to_mask(test_index, size=self.test_data.x.size(0))
            if torch_geometric.__version__=='1.7.0':
                self.__data_list__[0] = data
            else:
                self._data_list[0] = data
        self.train_index = torch.tensor(train_index)
        self.val_index = torch.tensor(val_index)
        self.test_index = torch.tensor(test_index)
        print(f'Train on {self.cur_dataset}. Test on {self.test_dataset}')
        print(f'Number of edges: {self[0].edge_index.size(1)}')
        print(f'Number of nodes: {self[0].x.size(0)}')
        print(f'Number of labeled nodes: {len(labels)}')
        print(f'Number of social bots: {sum(labels)}')
        print(f'Number of training nodes: {len(self.train_index)}')
        print(f'Number of test nodes: {len(self.test_index)}')

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = True
    return mask