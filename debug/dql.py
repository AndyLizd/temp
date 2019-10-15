# Deep Q-Learning

import numpy as np 
import torch
# from model import QNet
import gym
from pdb import set_trace as bk
from lib import e_greedy, Buffer
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=20):
        super(QNet, self).__init__()
        self.Q_val = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hid_size, act_size),
            # nn.Tanh(),
        )
    
    def forward(self, x):
        return self.Q_val(x)

# parameters
LEARNING_RATE = 1e-4
gamma = 1.0
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
greed_after_n = 800
n_episode = 5000
num_per_batch = 30
buffer_maxlen = 800
n_update_target_net = 30
hid_size = 80
ENV_NAME = ['CartPole-v1','LunarLander-v2'][0]
seed = 100

# best testing reward
best_test_r = 0
PATH_TO_SAVE = './saved/dql_'+ ENV_NAME +'.pth'

env = gym.make(ENV_NAME)
env.seed(seed)

num_obs = env.observation_space.shape[0]
num_act = env.action_space.n 
# neural netword
# initialize action-value function Q
qnet = QNet(num_obs, num_act, hid_size=hid_size)
qnet_target = QNet(num_obs, num_act, hid_size=hid_size)
qnet_target.load_state_dict(qnet.state_dict())

LossMSE = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(qnet.parameters(), lr=LEARNING_RATE)

# initialize replay buffer memory with capacity maxlen
buffer = Buffer(maxlen=buffer_maxlen)

# training
for i_eps in range(n_episode):
    s1 = env.reset()
    total_reward = 0
    while True:
        epsilon = max(epsilon_min, epsilon_decay*epsilon)
        act = e_greedy(s1, qnet, env, epsilon)
        s2, r, done, _ = env.step(act)
        total_reward += r

        buffer.append({'s1':s1, 'act':act, 'r':r, 's2':s2, 'done':done})
        s1 = s2
        # if the buffer is full, start training
        if len(buffer) >= buffer_maxlen:
            samples = buffer.sample(num=num_per_batch)
            # Q target by one step bootstrapping
            q_labels = []
            for sample in samples:
                if sample['done']:
                    q_label = sample['r']
                else:
                    q_label = sample['r'] + gamma*max(qnet_target(torch.Tensor(sample['s2']))).detach()
                q_labels.append(q_label)
            q_labels = torch.tensor(q_labels, dtype=torch.float)
            # current Q estimation
            s1_v = torch.tensor([sample['s1'] for sample in samples], dtype=torch.float)
            act_v = torch.tensor([sample['act'] for sample in samples])
            q_current = qnet(s1_v).gather(1, act_v.unsqueeze(1)).squeeze(1)

            loss = LossMSE(q_labels, q_current)
            qnet.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            qnet.eval()



        if done:
            # testing the model
            if i_eps%50 == 0:
                test_rs = []
                for i in range(5):
                    test_r = 0
                    obs = env.reset()
                    while True:
                        act = e_greedy(obs, qnet, env, epsilon=0)
                        obs, r, done, _ = env.step(act)
                        test_r += r
                        if done:
                            test_rs.append(test_r)
                            break
                average_r = sum(test_rs)/len(test_rs)
                print('ep:', i_eps, ' total_r:', average_r)

                if average_r >= best_test_r:
                    best_test_r = average_r
                    # torch.save(qnet, PATH_TO_SAVE)
                    torch.save({'state_dict': qnet.state_dict()}, PATH_TO_SAVE)
                    print('Find and save a better model!')

            # update the target network
            if i_eps%n_update_target_net == 0:
                qnet_target.load_state_dict(qnet.state_dict())
            break

# load model
# model = torch.load(PATH_TO_SAVE)
# model.eval()

bk()