import os.path as osp
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sys import argv
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from gym.spaces import Discrete
from gym import spaces
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
from model.SocialData import SocialBotDataset
from sklearn import preprocessing
import time
random.seed(0)
torch.manual_seed(0)
# dataset = dataset.shuffle()
general=True
dataset_str = "cresci-2015"
test_dataset = "varol-2017"
#{"cresci-2015":0,"botometer-feedback-2019":1,"cresci-rtbust-2019":2,"vendor-purchased-2019":4,"varol-2017":5}


class Net(torch.nn.Module):
    def __init__(self, max_layer=10, dataset='Cora'):
        self.hidden = []
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        for i in range(max_layer - 2):
            self.hidden.append(GCNConv(16, 16, cached=True))
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, action, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for i in range(action - 2):
            x = F.relu(self.hidden[i](x, edge_index))
            x = F.dropout(x, training=self.training)
        self.embedding = self.conv2(x, edge_index)
        return F.log_softmax(self.embedding, dim=1)


class gcn_env(object):
    def __init__(self, dataset='Cora', lr=0.01, weight_decay=5e-4, max_layer=10, batch_size=128, policy="",K=0,general=general,test_dataset=test_dataset):
        device = 'cpu'
        dataset = dataset
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = SocialBotDataset(root="./data", pre_transform=min_max_scaler.fit_transform,K=K,General=general,dataset=dataset,test_dataset=test_dataset)
        data = dataset[0]
        self.dataset = dataset
        # self.test_data = self.dataset.test_data
        # adj = to_dense_adj(data.edge_index).numpy()[0]
        # norm = np.array([np.sum(row) for row in adj])
        # self.adj = (adj / norm).T
        self.ini_k_hop_target_user(data,max_layer)

        self.model, self.data = Net(max_layer, dataset).to(device), data.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        # train_mask = self.data.train_mask.to('cpu').numpy()
        self.train_indexes = self.dataset.train_index
        self.batch_size = len(self.train_indexes) - 1
        self.i = 0
        self.val_acc = 0.0
        self._set_action_space(max_layer)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.max_layer = max_layer

        # For Experiment #
        self.random = False
        self.gcn = False  # GCN Baseline
        self.enable_skh = True  # only when GCN is false will be useful
        self.enable_dlayer = True
        self.baseline_experience = 50

        # buffers for updating
        # self.buffers = {i: [] for i in range(max_layer)}
        self.buffers = defaultdict(list)
        self.past_performance = [0]

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def init_k_hop(self, max_hop):
        sp_adj = csr_matrix(self.adj)
        dd = sp_adj
        self.adjs = [dd]
        for i in range(max_hop):
            dd *= sp_adj
            self.adjs.append(dd)

    def ini_k_hop_target_user(self,data,max_hop):
        print("ini k hop neighbor")
        sp_adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
        sg_num = self.dataset.data.y.shape[0]
        self.adjs = []
        dd = sp_adj[:sg_num, :]
        target_adj = dd[:, sg_num]
        target_adj = target_adj.toarray()
        target_adj = normalize(target_adj, norm='l1', axis=1)
        self.adjs.append(target_adj)
        for hop in range(max_hop):
            dd = dd * sp_adj
            target_adj = dd[:, :sg_num]
            target_adj = target_adj.toarray()
            target_adj = normalize(target_adj, norm='l1', axis=1)
            self.adjs.append(target_adj)
            print(f"done {hop} hop.")

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.model.train()
        self.optimizer.zero_grad()
        if self.random == True:
            action = random.randint(1, 5)
        # train one step
        index = self.train_indexes[self.i]
        pred = self.model(action, self.data)[index]
        pred = pred.unsqueeze(0)
        y = self.data.y[index]
        y = y.unsqueeze(0)
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

        # get reward from validation set
        val_acc = self.eval_batch()

        # get next state
        self.i += 1
        self.i = self.i % len(self.train_indexes)
        next_index = self.train_indexes[self.i]
        # next_state = self.data.x[next_index].to('cpu').numpy()
        next_state = self.data.x[next_index].numpy()
        if self.i == 0:
            done = True
        else:
            done = False
        return next_state, val_acc, done, "debug"

    def reset2(self):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def step2(self, actions):
        self.model.train()
        self.optimizer.zero_grad()
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        done = False
        for act, idx in zip(actions, index):
            if self.gcn == True or self.enable_dlayer == False:
                act = self.max_layer
            self.buffers[act].append(idx)
            if len(self.buffers[act]) >= self.batch_size:
                self.train(act, self.buffers[act])
                self.buffers[act] = []
                done = True
        if self.gcn == True or self.enable_skh == False:
            ### Random ###
            self.i += min((self.i + self.batch_size) % self.batch_size, self.batch_size)
            start = self.i
            end = (self.i + self.batch_size) % len(self.train_indexes)
            index = self.train_indexes[start:end]
        else:
            index = self.stochastic_k_hop(actions, index)
        next_state = self.data.x[index].to('cpu').numpy()
        # next_state = self.data.x[index].numpy()
        val_acc_dict = self.eval_batch()
        val_acc = [val_acc_dict[a] for a in actions]
        test_acc = self.test_batch()
        baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
        self.past_performance.extend(val_acc)
        reward = [100 * (each - baseline) for each in val_acc]  # FIXME: Reward Engineering
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)
        return next_state, reward, [done] * self.batch_size, (val_acc, r)

    def stochastic_k_hop(self, actions, index):
        next_batch = []
        for idx, act in zip(index, actions):
            prob = self.adjs[act][idx]
            prob = prob if np.sum(prob) > 0. else np.full(len(prob), 1. / len(prob))
            cand = np.array([i for i in range(len(prob))])
            next_cand = np.random.choice(cand, p=prob)
            next_batch.append(next_cand)
        return next_batch

    def train(self, action, indexes):
        self.model.train()
        pred = self.model(action, self.data)[indexes]
        y = self.data.y[indexes]
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

    def eval_batch(self):
        self.model.eval()
        batch_dict = {}
        val_index = self.dataset.val_index
        val_states = self.data.x[val_index].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(val_index))
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(val_index), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(val_index, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        # acc = 0.0
        acc = {a: 0.0 for a in range(self.max_layer)}
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.data)
            pred = logits[idx].max(1)[1]
            # acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
            acc[a] = pred.eq(self.data.y[idx]).sum().item() / len(idx)
        # acc = acc / len(batch_dict.keys())
        return acc

    def test_batch(self):
        self.model.eval()
        batch_dict = {}
        # test_index = np.where(self.data.test_mask.to('cpu').numpy() == True)[0]
        test_index = self.dataset.test_index
        val_states = self.dataset.test_data.x[test_index].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(test_index))
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(test_index), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(test_index, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        acc = 0.0
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.dataset.test_data)
            pred = logits[idx].max(1)[1]
            acc += pred.eq(self.dataset.test_data.y[idx]).sum().item() / len(idx)
        acc = acc / len(batch_dict.keys())
        return acc

    def check(self):
        self.model.eval()
        train_index = np.where(self.data.train_mask.to('cpu').numpy() == True)[0]
        tr_states = self.data.x[train_index].to('cpu').numpy()
        tr_acts = self.policy.eval_step(tr_states)

        val_index = np.where(self.data.val_mask.to('cpu').numpy() == True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()
        val_acts = self.policy.eval_step(val_states)

        test_index = np.where(self.data.test_mask.to('cpu').numpy() == True)[0]
        test_states = self.data.x[test_index].to('cpu').numpy()
        test_acts = self.policy.eval_step(test_states)

        return (train_index, tr_states, tr_acts), (val_index, val_states, val_acts), (
        test_index, test_states, test_acts)

random.seed(0)
torch.manual_seed(0)


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Normalizer(object):
    ''' Normalizer class that tracks the running statistics for normlization
    '''

    def __init__(self):
        ''' Initialize a Normalizer instance.
        '''
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 1000
        self.length = 0

    def normalize(self, s):
        ''' Normalize the state with the running mean and std.

        Args:
            s (numpy.array): the input state

        Returns:
            a (int):  normalized state
        '''
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        ''' Append a new state and update the running statistics

        Args:
            s (numpy.array): the input state
        '''
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.mean(self.state_memory, axis=0)
        self.length = len(self.state_memory)
class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        if len(self.memory) < self.batch_size:
            samples = random.sample(self.memory, len(self.memory))
        else:
            samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

class DQNAgent(object):
    def __init__(self,
                 scope,
                 replay_memory_size=2000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.2,
                 epsilon_decay_steps=100,
                 batch_size=128,
                 action_num=2,
                 state_shape=None,
                 norm_step=100,
                 mlp_layers=None,
                 learning_rate=0.0005,
                 device=None):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            scope (str): The name of the DQN agent
            env (object): The Environment.
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sampel when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (int): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (int): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            action_num (int): The number of the actions
            state_space (list): The space of the state vector
            norm_step (int): The number of the step used form noramlize state
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
        '''
        self.scope = scope
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.norm_step = norm_step

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        #with tf.variable_scope(scope):
        self.q_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)
        self.target_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)

        # Create normalizer
        self.normalizer = Normalizer()

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)


    def learn(self, env, total_timesteps):
        done = [False]
        next_state_batch = env.reset2()
        trajectories = []
        for t in range(total_timesteps):
            A = self.predict_batch(next_state_batch)
            best_actions = np.random.choice(np.arange(len(A)), p=A, size=next_state_batch.shape[0])
            state_batch = next_state_batch
            next_state_batch, reward_batch, done_batch, debug = env.step2(best_actions) # debug = (val_acc, test_acc)
            trajectories = zip(state_batch, best_actions, reward_batch, next_state_batch, done_batch)
            for each in trajectories:
                self.feed(each)
        # print(len(trajectories))
        loss = self.train()
        return loss, reward_batch, debug

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the Normalizer to calculate mean and std.
            The transition is NOT stored in the memory
            In stage 2, the transition is stored to the memory.

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        if self.total_t < self.norm_step:
            self.feed_norm(state, action, reward, next_state, done)
        else:
            self.feed_memory(state, action, reward, next_state, done)
        self.total_t += 1

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        A = self.predict(state)
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def eval_step(self, states):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        q_values = self.q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_actions = np.argmax(q_values, axis=1)
        return best_actions

    def predict(self, state):
        ''' Predict the action probabilities but have them
            disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator.predict_nograd(np.expand_dims(self.normalizer.normalize(state), 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def predict_batch(self, states):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action = np.argmax(q_values, axis=1)
        for a in best_action:
            A[best_action] += (1.0 - epsilon)
        A = A/A.sum()
        return A

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN)
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(len(best_actions)), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)

        self.train_t += 1
        return loss

    def feed_norm(self, state, action, reward, next_state, done):
        ''' Feed state to normalizer to collect statistics

        Args:
            state (numpy.array): the state that will be feed into normalizer
        '''
        self.normalizer.append(state)
        self.memory.save(self.normalizer.normalize(state), action, reward, self.normalizer.normalize(next_state), done)

    def feed_memory(self, state, action, reward, next_state, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        '''
        self.memory.save(self.normalizer.normalize(state), action, reward, self.normalizer.normalize(next_state), done)

class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, action_num=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
        ''' Initilalize an Estimator object.

        Args:
            action_num (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.action_num = action_num
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(action_num, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).to('cpu').numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, action_num)
        q_as = self.qnet(s)

        # (batch, action_num) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss

class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, action_num=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            action_num (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.action_num, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)


def main(K=0):
    torch.backends.cudnn.deterministic=True
    ### Experiment Settings ###
    # Cora
    max_timesteps = 1
    dataset = dataset_str
    print(dataset)
    max_episodes = 20
    ### Experiment Settings ###

    env = gcn_env(dataset=dataset, max_layer=3,K=K)
    env.seed(0)
    agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=200,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    device=torch.device('cpu')
            )
    start = time.time()
    env.policy = agent
    last_val = 0.0
    # Training: Learning meta-policy
    print("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = agent.learn(env, max_timesteps) # debug = (val_acc, reward)
        if val_acc > last_val: # check whether gain improvement on validation set
            best_policy = deepcopy(agent) # save the best policy
        last_val = val_acc
        print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward)

    # Testing: Apply meta-policy to train a new GNN
    test_acc = 0.0
    print("Training GNNs with learned meta-policy")
    end = time.time()
    print(f" policy training time: {end - start}")
    new_env = gcn_env(dataset=dataset, max_layer=3,K=K)
    start = time.time()
    new_env.seed(0)
    new_env.policy = best_policy
    accs = []
    state = new_env.reset2()
    for i_episode in range(1, 31):
        action = best_policy.eval_step(state)
        state, reward, done, (val_acc, reward) = new_env.step2(action)
        test_acc = new_env.test_batch()
        accs.append(test_acc)
        print("Training GNN", i_episode, "; Val_Acc:", val_acc, "; Test_Acc:", test_acc)
    end = time.time()
    print(f" GNN training time: {end - start}")
    return max(accs)

if __name__ == "__main__":
    K = 10
    test_accs = []
    for item in range(K):
        print(f"Start {item} fold")
        test_acc = main(item)
        test_accs.append(test_acc)
        print(test_accs)
    print(f"RoSGAS test ACC mean: {np.mean(test_accs)} , var: {np.var(test_accs)}")