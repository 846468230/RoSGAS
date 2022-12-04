import os
import sys
import numpy
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import argparse
from model.gcn_exp import gcn_env
from model.dqn_agent import QAgent
from copy import deepcopy
import numpy as np
torch.manual_seed(12345)
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

parser = argparse.ArgumentParser(description='RoSGAS')
parser.add_argument('--dataset', type=str, default="varol-2017")
parser.add_argument('--general', type=bool, default=False)
parser.add_argument('--test_dataset', type=str, default="botometer-feedback-2019")
parser.add_argument('--redirect', type=bool, default=False)
#{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}
parser.add_argument('--folds', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_timesteps', type=int, default=1)
parser.add_argument('--max_episodes', type=int, default=100)

parser.add_argument('--replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_init_size', type=int, default=500)
parser.add_argument('--update_target_estimator_every', type=int, default=1)

parser.add_argument('--layer_num', type=int, default=3)
parser.add_argument('--width_num',type=int,default=2)
parser.add_argument('--discount_factor', type=float, default=0.95)
parser.add_argument('--epsilon_start', type=float, default=1.)
parser.add_argument('--epsilon_end', type=float, default=0.1)
parser.add_argument('--epsilon_decay_steps', type=int, default=100)
parser.add_argument('--norm_step', type=int, default=200)
parser.add_argument('--mlp_layers', type=list, default=[64, 128, 256, 128, 64])

parser.add_argument('--sg_encoder', type=str, default='GCN')
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=2)
args = parser.parse_args()
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if args.redirect:
    if args.general:
        sys.stdout = open(f'{args.dataset}{args.test_dataset}.txt', "w")
    else:
        sys.stdout = open(f'{args.dataset}{args.width_num}{args.layer_num}.txt', "w")

print(f"Searching with {args.layer_num} layers and {args.width_num} width")
def main(K=0):
    start = time.time()
    env = gcn_env(dataset = args.dataset, folds = args.folds,
                  max_layer = args.layer_num,
                  max_width = args.width_num,
                  hid_dim = args.hid_dim, out_dim = args.out_dim,
                  lr = args.lr, weight_decay = args.weight_decay,
                  device = args.device,
                  policy = "",K=K,general=args.general,test_dataset=args.test_dataset)
    agent = QAgent(replay_memory_size = args.replay_memory_size,
                   replay_memory_init_size = args.replay_memory_init_size,
                   update_target_estimator_every = args.update_target_estimator_every,
                   discount_factor = args.discount_factor,
                   epsilon_start = args.epsilon_start,
                   epsilon_end = args.epsilon_end,
                   epsilon_decay_steps = args.epsilon_decay_steps,
                   lr=args.lr,
                   batch_size=env.batch_size_qdn,
                   sg_num = env.sg_num,
                   layer_num=env.max_layer,
                   width_num=env.width_num,
                   norm_step=args.norm_step,
                   mlp_layers=args.mlp_layers,
                   state_shape=env.state_shape,
                   device=args.device)
    env.policy = agent
    end = time.time()
    print(f"ini time: {end - start}")
    start = time.time()
    last_val = 0.0
    # Training: Learning with RL agent
    print("Training RL agent on Validation Set")
    for i_episode in range(1, args.max_episodes + 1):
        loss1,loss2, _, (val_acc, mean_reward) = agent.learn(env, args.max_timesteps)
        if val_acc >= last_val:
            best_policy = deepcopy(agent)
        last_val = val_acc
        print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", mean_reward)
    end = time.time()
    print(f"agent training time: {end - start}")

    print("Training GNNs with learned RL agent")
    # del env,agent
    # gc.collect()
    # if torch.cuda.is_available():
    # 	best_policy.transfer_to_cpu()
    new_env = gcn_env(dataset = args.dataset, folds = args.folds,
                      max_layer=args.layer_num,
                      max_width=args.width_num,
                      hid_dim = args.hid_dim, out_dim = args.out_dim,
                      lr = args.lr, weight_decay = args.weight_decay,
                      device = args.device,
                      policy = "",K=K,general=args.general,test_dataset=args.test_dataset)
    new_env.policy = best_policy
    start = time.time()
    states = new_env.reset()
    accs = []
    home_scores = []
    for i_episode in range(1, 31):
        actions = new_env.policy.eval_step(states)
        states, rewards, dones, (val_acc, mean_reward) = new_env.step(actions)
        test_acc,labels,features = new_env.test()
        accs.append(test_acc)
        # home_scores.append(home_score)
        print("Training GNN", i_episode, "; Val ACC:", val_acc, "; Test ACC:", test_acc)
        if i_episode==30:
            features = prepocessing_tsne(features,2)
            plot_figure(features,labels,"RoSGAS_"+new_env.dataset.cur_dataset)
    end = time.time()
    print(f"GNN training time: {end - start}")
    return max(accs),max(home_scores)

if __name__ == '__main__':
    K = 10
    test_accs = []
    hom_scores = []
    for item in range(K):
        print(f"Start {item} fold")
        test_acc,hom_score = main(item)
        test_accs.append(test_acc)
        hom_scores.append(hom_score)
        print(test_accs)
    print(f"RoSGAS test ACC mean: {np.mean(test_accs)} , var: {np.var(test_accs)}")
    print(f"RoSGAS homogeneity score mean: {np.mean(hom_scores)} , var: {np.var(hom_scores)}")
