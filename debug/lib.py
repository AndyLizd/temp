# library of helper functions
import numpy as np
import torch
import collections
import gym
from pdb import set_trace as bk

seed = 123
np.random.seed(seed)

# choose epsilon based on a ramp function
def choose_epsilon(epsilon_bd, i_eps, n_episode, greed_after_n=-1):
    # greed_after_n: choose the smallest epsilon after n
    if greed_after_n == -1:
        greed_after_n = int(n_episode*0.5)
    delta_epsilon = epsilon_bd[1] - epsilon_bd[0]
    epsilon = epsilon_bd[1] - delta_epsilon*i_eps/greed_after_n
    return max(epsilon, epsilon_bd[0])


# epsilon greedy selection of actions
def e_greedy(obs, net, env, epsilon= 0.05):
    if np.random.random() > epsilon:
        q_vals = net(torch.FloatTensor(obs))
        act = torch.argmax(q_vals).item()
    else:
        act = env.action_space.sample()
    return act

# replay buffer
class Buffer:
    def __init__(self, maxlen = 100):
        self.buffer = collections.deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def append(self, sarsd_tuple):
        self.buffer.append(sarsd_tuple)

    def sample(self, num):
        return np.random.choice(self.buffer, num, replace=False)

# policy gradient trajectory
class Traj:
    def __init__(self, gamma):
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.rs, self.Rt, self.Adv, self.Log_a = [], [], [], []
        self.obs_v, self.baseline = [], []

    def append(self, s1, r, log_a):
        self.obs_v.append(s1)
        self.rs.append(r)
        self.Log_a.append(log_a)

    def process(self, v_net):
        self.Rt = [self.rs.pop()]
        for r in reversed(self.rs):
            self.Rt.append(self.gamma*self.Rt[-1]+r)
        self.Rt.reverse()
        self.Rt = torch.tensor(self.Rt).type(torch.float)
        self.obs_v = np.array(self.obs_v)
        self.obs_v = torch.from_numpy(self.obs_v).type(torch.float)
        self.baseline = v_net(self.obs_v).squeeze(1)
        self.Adv = self.Rt - self.baseline.detach()
        self.Log_a = torch.stack(self.Log_a)#.type(torch.float)

        return self.Rt, self.baseline, self.Adv, self.Log_a

# update a neural net
def update_net(net, loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Wrapper of ENV
class ObsScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, low=[], high=[]):
        super(ObsScaleWrapper, self).__init__(env)
        if low == [] and high == []:
            self.low = self.observation_space.low
            self.high = self.observation_space.high
        else:
            self.low = low
            self.high = high
        self.norm_ftr = (self.high - self.low)
        if any(i<=0 for i in self.norm_ftr):
            print('a lower bound is larger than the high bound')
            raise EnvironmentError

    def observation(self, obs):
        return (obs-self.low)/self.norm_ftr
