# vanilla policy gradient
import torch
from torch.distributions import Categorical
import numpy as np 
import gym
from model import V_Net, ActorNet
from lib import Traj, update_net

from pdb import set_trace as bk

# parameters
ENV_NAME = ['CartPole-v1','LunarLander-v2'][0]
seed = 10
vnet_hid_size = 30
act_hid_size = 80
n_episode = 100000
gamma = 0.99
CRITIC_LR = 1e-3
ACTOR_LR = 1e-2

env = gym.make(ENV_NAME)
env.seed(seed)
torch.manual_seed(seed)

num_obs = env.observation_space.shape[0]
num_act = env.action_space.n

# Init Neural Net and Optimizer
critic = V_Net(num_obs, hid_size=vnet_hid_size)
actor = ActorNet(num_obs, num_act, hid_size=act_hid_size)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr = CRITIC_LR)
optimizer_actor = torch.optim.Adam(actor.parameters(), lr = ACTOR_LR)
LossMSE = torch.nn.MSELoss(reduction='sum')



# for testing
average_total_r = 0


for i_eps in range(n_episode):
    # init trajectory
    traj = Traj(gamma)
    # init env
    s1 = env.reset()
    total_r = 0
    while True:
        logit = actor(torch.from_numpy(s1).type(torch.float))
        logit_c = Categorical(logit)
        act = logit_c.sample()
   
        s2, r, done, _ = env.step(act.item())
        total_r += r
        s1 = s2
        # collect and append information 
        log_a = logit_c.log_prob(act)
        traj.append(s1, r, log_a)
        if done:
            average_total_r = 0.05*total_r + average_total_r*0.95 
            if i_eps%50 == 0:
                print(i_eps, average_total_r)
            break

        
    Rt, baseline, Adv, Log_a = traj.process(critic)
    # update critic network
    critic_loss = LossMSE(baseline, Rt)
    update_net(critic, critic_loss, optimizer_critic)

    # update actor network
    actor_loss = -torch.mean(torch.mul(Log_a, Adv))
    update_net(actor, actor_loss, optimizer_actor)

    bk()

    del traj
    
   
