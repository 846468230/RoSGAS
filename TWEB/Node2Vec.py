import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
import numpy as np

def main(K=0):
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform,
                               transer_y=True,K=K)  # transform=NormalizeFeatures())#pre_transform= min_max_scaler.fit_transform)
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()
    data = dataset[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=10,
                     context_size=2, walks_per_node=2,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=min(len(dataset.train_index),len(dataset.val_index),len(dataset.test_index)), shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=100)
        return acc
    accs = []
    for epoch in range(1, 31):
        loss = train()
        acc = test()
        accs.append(acc)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(data.y.size(0), device=device))
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(dataset.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
    # plot_points(colors)
    return max(accs)


if __name__ == "__main__":
    K = 10
    test_accs = []
    for item in range(K):
        print(f"Start {item} fold")
        test_acc = main(item)
        test_accs.append(test_acc)
        print(test_accs)
    print(f"Node2Vec Mean: {np.mean(test_accs)} , Var: {np.var(test_accs)}")
