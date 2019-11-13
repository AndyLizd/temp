import numpy as np 
from pdb import set_trace as bk

# given state x and action, sample the next state x_next
def env_sample_next_step(act, x):
    x_next = [x[0]+1, x[1]-1]
    return x_next

# calculate Pr(obs|x) ~ gaussion
def obs_prob(obs, x):
    return 0.5

# calculate reward (one step) given observation
def cal_reward(obs):
    R = 0
    return R

# particle filter: given current belief, action, and next observation, return the next belief
def par_filter(theta, act, obs_next, N = 50):
    theta_next = {'x':[], 'w':[]}
    for _ in range(N):
        pick_idx = np.random.choice(range(len(theta['x'])), p=theta['w'])
        x = theta['x'][pick_idx]        
        x_next = env_sample_next_step(act, x)
        w_next = obs_prob(obs_next, x_next)
        theta_next['x'].append(x_next)
        theta_next['w'].append(w_next)
    theta_next['w'] = 1/sum(theta_next['w'])*np.array(theta_next['w']) 
    return theta_next

# given current current belief, action, returns sets of possible beliefs
def par_projection(theta, act, N_thetas = 50, N_pf = 10):
    Thetas = ['theta':[], 'R':[]]
    for _ in range(N_thetas):
        pick_idx = np.random.choice(range(len(theta['x'])), p=theta['w']))
        x = theta['x'][pick_idx]        
        obs_next = ?????
        theta_next = par_filter(theta, act, obs_next, N=N_pf)
        Thetas['theta'].append(theta_next)
        R_next = cal_reward(obs_next)
        Thetas['R'].append(R_next)

if __name__ == "__main__":
    theta = {'x':[[1,2], [5,0], [0,0]], 'w':[0.5,0.3,0.2]}
    act = 1.1
    obs_next = [3,2]

    theta_next = par_filter(theta, act, obs_next, N=3)

    print(theta_next)

