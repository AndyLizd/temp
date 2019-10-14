import torch
import torch.nn as nn
import numpy as np 
from pdb import set_trace as bk

# Q network - states -> Q values
class Q_Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=20):
        super(Q_Net, self).__init__()
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

# V network - states -> Value
class V_Net(nn.Module):
    def __init__(self, obs_size, hid_size=20):
        super(V_Net, self).__init__()
        self.V_val = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.V_val(x)

# Actor network - states -> actions
class ActorNet(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=20):
        super(ActorNet, self).__init__()
        self.act = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, act_size),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        return self.act(x)


if __name__ == '__main__':
    batch_n = 5
    x = torch.rand(batch_n,10)
    y_label = torch.rand(batch_n,5)
    
    num_obs = x.size()[1]
    num_act = y_label.size()[1]

    learning_rate = 1e-4

    net = Q_Net(num_obs, num_act)
    net2 = Q_Net(num_obs, num_act)
    LossMSE = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for t in range(5000):
        y_pred = net.forward(x)
        bk()
        loss = LossMSE(y_pred, y_label)
        # loss = sum(sum(abs(y_label-y_pred)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t%100 == 0:
            print(t, loss.item())


    # copy weight
    net2.load_state_dict(net.state_dict())
    
    bk()