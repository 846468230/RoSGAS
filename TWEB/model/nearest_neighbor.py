import numpy as np
from scipy.spatial import cKDTree

class NearestNeighbor:
    def __init__(self,planning_horizon,L,discount,buf_max_size,K_neighbors,r_scale):

        # parameters to tune
        self.horizon = planning_horizon
        self.L = L
        self.discount = discount

        self.max_size = buf_max_size
        self.states_all = []
        self.storage = []
        self.ptr = 0

        self.K_neighbors = K_neighbors
        self.r_scale = r_scale
        self.tree = None


    def remember(self, state, action, reward, next_state, step, terminal):
        if len(self.storage) == self.max_size:
            self.states_all[self.ptr] = np.concatenate((state,np.array([action])))
            self.storage[self.ptr] = (state, action, self.r_scale * reward, next_state, step, terminal)
            self.ptr = int((self.ptr + 1) % self.max_size)
        else:
            b = np.array(action)
            self.states_all.append(np.concatenate((state, np.array([action]))))
            self.storage.append((state, action, self.r_scale * reward, next_state, step, terminal))
            
            
    def estimate(self, step, s, a = None):
        if step == self.horizon:
            return 0

        distances, indices = self.tree.query(
            np.concatenate((s,np.array([a]))), k = self.K_neighbors, n_jobs = -1)
        if distances == np.inf:
            return 0
        nearest_neighbors = self.storage[indices]

        vals = []
        for i in range(self.K_neighbors):
            nn = nearest_neighbors[i] if self.K_neighbors > 1 else nearest_neighbors
            d = distances[i] if self.K_neighbors > 1 else distances
            if nn[-1]:
                vals.append(nn[2] + self.L * d)
            else:
                vals.append(nn[2] + self.L * d + self.discount * self.estimate(step + 1, nn[3]))

        return np.min(vals)
        

    def estimate_batch(self,step,s_batch,a_batch):
        estimate_values = []
        for i in range(s_batch.shape[0]):
            v_hat = self.estimate(step,s_batch[i].cpu(),a_batch[i].cpu())
            estimate_values.append(v_hat)
        return np.array(estimate_values)

    def learn(self, s, a, r, s_, step):
        v = self.estimate(step, s)
        v_ = self.estimate(step + 1, s_)
        td_error = r + self.discount * v_ - v

        return td_error if td_error > 0 else td_error * self.config.neg_td_scale
