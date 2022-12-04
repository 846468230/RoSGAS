import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random
from .nearest_neighbor import NearestNeighbor
from scipy.spatial import cKDTree
from sklearn import preprocessing
from sklearn.metrics import f1_score

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)
np.random.seed(12345)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class Normalizer(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 1000
        self.length = 0

    def normalize(self, s):
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.std(self.state_memory, axis=0)
        self.length = len(self.state_memory)


class Memory(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        if len(self.memory) < self.batch_size:
            samples = random.sample(self.memory, len(self.memory))
        else:
            samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))


class QAgent(object):
    def __init__(self,
                 replay_memory_size, replay_memory_init_size, update_target_estimator_every,
                 discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps,
                 lr, batch_size,
                 sg_num,
                 layer_num,
                 width_num,
                 norm_step,
                 mlp_layers,
                 state_shape,
                 device,
                 search_width=True):
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.sg_num = sg_num
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.width_num = width_num
        self.norm_step = norm_step
        self.device = device

        self.total_t = 0
        self.train_t = 0
        self.episode = 0
        self.layer_q_estimator = Estimator(action_num=layer_num,
                                           lr=lr,
                                           state_shape=self.state_shape,
                                           mlp_layers=mlp_layers,
                                           device=device)
        self.layer_target_estimator = Estimator(action_num=layer_num,
                                                lr=lr,
                                                state_shape=self.state_shape,
                                                mlp_layers=mlp_layers,
                                                device=self.device)
        self.layer_memory = Memory(replay_memory_size, batch_size)
        if search_width:
            self.width_q_estimator = Estimator(action_num=width_num,
                                               lr=lr,
                                               state_shape=self.state_shape,
                                               mlp_layers=mlp_layers,
                                               device=device)
            self.width_target_estimator = Estimator(action_num=width_num,
                                                    lr=lr,
                                                    state_shape=self.state_shape,
                                                    mlp_layers=mlp_layers,
                                                    device=self.device)
            self.width_memory = Memory(replay_memory_size, batch_size)
        self.normalizer = Normalizer()

    def transfer_to_cpu(self):
        self.device = torch.device('cpu')
        self.layer_q_estimator.device = self.device
        self.layer_target_estimator.device = self.device
        self.width_q_estimator.device = self.device
        self.width_target_estimator.device = self.device
        self.layer_q_estimator.qnet.to(self.device)
        self.layer_target_estimator.qnet.to(self.device)
        self.width_q_estimator.qnet.to(self.device)
        self.width_target_estimator.qnet.to(self.device)
        torch.cuda.empty_cache()

    def learn(self, env, total_timesteps):
        next_states = env.reset()
        self.episode+=1
        trajectories = []
        for t in range(total_timesteps):
            A_1, A_2 = self.predict_batch_new(next_states)
            states = next_states

            next_states, rewards, dones, debug = env.step((A_1, A_2))
            trajectories = zip(states, A_1, A_2, rewards, next_states, dones)
            for ts in trajectories:
                self.feed(ts)
        loss1, loss2 = self.train()
        return loss1, loss2, rewards, debug

    def feed(self, ts):
        (state, action1, action2, reward, next_state, done) = tuple(ts)
        if self.total_t < self.norm_step:
            self.feed_norm(state)
            self.feed_memory(state, action1, action2, reward, next_state, done)
        else:
            self.feed_memory(state, action1, action2, reward, next_state, done)
        self.total_t += 1

    def eval_step(self, states):
        q1_values = self.layer_q_estimator.predict_nograd(states)
        best_action1s = np.argmax(q1_values, axis=-1)
        q2_values = self.width_q_estimator.predict_nograd(states)
        best_action2s = np.argmax(q2_values, axis=-1)
        return (best_action1s, best_action2s)

    def predict_batch(self, states):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A_1 = np.ones(self.layer_num, dtype=float) * epsilon / self.layer_num
        q_1_values = self.layer_q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action_1 = np.argmax(q_1_values, axis=1)
        for a in best_action_1:
            A_1[best_action_1] += (1.0 - epsilon)
        A_1 = A_1 / A_1.sum()

        A_2 = np.ones(self.width_num, dtype=float) * epsilon / self.width_num
        q_2_values = self.width_q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action_2 = np.argmax(q_2_values, axis=1)
        for a in best_action_2:
            A_2[best_action_2] += (1.0 - epsilon)
        A_2 = A_2 / A_2.sum()
        best_action1s = np.random.choice(np.arange(len(A_1)), p=A_1, size=states.shape[0])
        best_action2s = np.random.choice(np.arange(len(A_2)), p=A_2, size=states.shape[0])
        return best_action1s, best_action2s

    def predict_batch_new(self, states):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A_1 = []
        q_1_values = self.layer_q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action_1 = np.argmax(q_1_values, axis=1)
        prob = np.random.random(states.shape[0])
        for i, a in enumerate(best_action_1):
            if prob[i] > epsilon:
                A_1.append(a)
            else:
                A_1.append(np.random.randint(0, self.layer_num))
        A_2 = []
        q_2_values = self.width_q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action_2 = np.argmax(q_2_values, axis=1)
        for i, a in enumerate(best_action_2):
            if prob[i] > epsilon:
                A_2.append(a)
            else:
                A_2.append(np.random.randint(0, self.width_num))
        return np.array(A_1), np.array(A_2)

    def train(self):
        state_batch, action1_batch, reward_batch, next_state_batch, done_batch = self.layer_memory.sample()
        q_1_values_next = self.layer_q_estimator.predict_nograd(next_state_batch)
        best_action1s = np.argmax(q_1_values_next, axis=1)

        q_1_values_next_target = self.layer_target_estimator.predict_nograd(next_state_batch)

        target1_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.discount_factor * q_1_values_next_target[np.arange(len(best_action1s)), best_action1s]

        state_batch = np.array(state_batch)
        loss1 = self.layer_q_estimator.update(state_batch, action1_batch, target1_batch,self.episode)

        state_batch, action2_batch, reward_batch, next_state_batch, done_batch = self.width_memory.sample()
        q_2_values_next = self.width_q_estimator.predict_nograd(next_state_batch)
        best_action2s = np.argmax(q_2_values_next, axis=1)

        q_2_values_next_target = self.width_target_estimator.predict_nograd(next_state_batch)

        target2_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.discount_factor * q_2_values_next_target[np.arange(len(best_action2s)), best_action2s]

        state_batch = np.array(state_batch)
        loss2 = self.width_q_estimator.update(state_batch, action2_batch, target2_batch,self.episode)

        if self.train_t % self.update_target_estimator_every == 0:
            self.layer_target_estimator = deepcopy(self.layer_q_estimator)
            self.width_target_estimator = deepcopy(self.width_q_estimator)

        self.train_t += 1
        return loss1, loss2

    def feed_norm(self, state):
        self.normalizer.append(state)

    def feed_memory(self, state, action1, action2, reward, next_state, done):
        self.layer_memory.save(self.normalizer.normalize(state), action1, reward, self.normalizer.normalize(next_state),
                               done)
        self.width_memory.save(self.normalizer.normalize(state), action2, reward, self.normalizer.normalize(next_state),
                               done)
        self.layer_q_estimator.nearest_neighbor.remember(self.normalizer.normalize(state), action1, reward,
                                                         self.normalizer.normalize(next_state), 0, done)
        self.width_q_estimator.nearest_neighbor.remember(self.normalizer.normalize(state), action2, reward,
                                                         self.normalizer.normalize(next_state), 0, done)


class Estimator(object):
    def __init__(self,
                 action_num,
                 lr,
                 state_shape,
                 mlp_layers,
                 device):
        self.device = device
        qnet = EstimatorNetwork(action_num, state_shape, mlp_layers)
        qnet = qnet.to(device)
        self.qnet = qnet
        self.nearest_neighbor = NearestNeighbor(planning_horizon=12, L=7, discount=0.99, buf_max_size=100000,
                                                K_neighbors=1, r_scale=0.1)
        self.qnet.eval()
        self.alpha = 0.5
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

    def predict_nograd(self, states):
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(self.device)
            q_as = self.qnet(states).to('cpu').numpy()
        return q_as

    def update(self, s, a, y,episode):
        self.optimizer.zero_grad()
        self.nearest_neighbor.tree = cKDTree(self.nearest_neighbor.states_all)
        self.qnet.train()
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        Q_tau = self.nearest_neighbor.estimate_batch(0, s, a)
        y = (self.alpha**episode) * Q_tau + (1 - self.alpha**episode) * y
        y = torch.from_numpy(y).float().to(self.device)
        q_as = self.qnet(s)
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        Q_loss = self.mse_loss(Q, y)
        Q_loss.backward()
        self.optimizer.step()
        Q_loss = Q_loss.item()
        self.qnet.eval()
        return Q_loss


class EstimatorNetwork(nn.Module):
    def __init__(self,
                 action_num,
                 state_shape,
                 mlp_layers):
        super(EstimatorNetwork, self).__init__()

        # build the Q network
        layer_dims = [state_shape[-1]] + mlp_layers

        fc = [nn.Flatten()]
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], action_num, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, states):
        return self.fc_layers(states)
