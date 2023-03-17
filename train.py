import REM
import loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy, Precision, Recall
import torch.optim as optim
import argparse
import wandb
wandb.init(project='REM')
# import gym
# import highway_env
# import d3rlpy 
# from d3rlpy.dataset import MDPDataset
# from d3rlpy.algos import DiscreteCQL, DiscreteBC
# from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
# from d3rlpy.metrics.scorer import evaluate_on_environment
# from d3rlpy.metrics.scorer import td_error_scorer
# from d3rlpy.metrics.scorer import average_value_estimation_scorer
# from sklearn.model_selection import train_test_split
# from gym.spaces import Box
import numpy as np
# import gym.spaces as spaces
# from gym import ObservationWrapper
import os
from data_loader import Dataloader

def seed_everything(seed=50):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # d3rlpy.seed(50)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(50)

# class FlattenObservation(gym.ObservationWrapper):

#     def __init__(self, env: gym.Env):

#         super().__init__(env)
#         self.observation_space = spaces.flatten_space(env.observation_space)

#     def observation(self, observation):
  
#         return spaces.flatten(self.env.observation_space, observation)



# ## ENV & data set up 
# env = gym.make("highway-v0")
# env.config["lanes_count"] = 4
# env = FlattenObservation(env)
# write a data loader 
path_list=['../../highway-env_conda/Observations_300kDQN_100ksteps_4lanes-64hu.csv', 
    '../../highway-env_conda/actions_300kDQN_100ksteps_4lanes-64hu.csv', 
    '../../highway-env_conda/rewards_300kDQN_100ksteps_4lanes-64hu.csv',
     '../../highway-env_conda/terminals_300kDQN_100ksteps_4lanes-64hu.csv'
]

dataset = Dataloader(path_list)
# dataset = MDPDataset.load("../DqnPolicy-4lanes-gamma0.99-100ksteps-64hu.h5") # vs 2lanes 
train_set, test_set = torch.utils.data.random_split(dataset, [int(0.9*len(dataset)), int(0.1 * len(dataset))])
train_data_loader = DataLoader(dataset= train_set, batch_size = 5, shuffle= True ) # TODO: change shffle to True  num_workers2=
test_data_loader = DataLoader(dataset =test_set, batch_size = 5, shuffle = False )#, num_workers=2 

parser = argparse.ArgumentParser()
parser.add_argument('-ep' , type= int, default= 200, help= 'number of epochs')
parser.add_argument('-num_heads' , type= int, default= 100, help= 'number of heads') #we can add all the hyper parameters if needed (later task)
# state, act, random_coeff
args = parser.parse_args()
n_epochs = args.ep
n_heads = args.num_heads

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = REM.DQN_REM(input_dim=25,fc1_dim=256, fc2_dim=256, num_actions=5,num_heads=n_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
criterion = loss.loss_fn(gamma=0.99, delta=1.0)

def make_coeff(num_heads, batch_size):
    # this would make the alfa coefficients 
    
    arr = np.random.uniform(low=0.0, high=1 , size= (batch_size,1, num_heads))  #B X 1xnum_heads
    arr /= np.sum(arr,axis=2, keepdims= True) # Bx1xN_heads
    # arr = np.repeat(arr,n_actions)
    return arr 


for epochs in range(n_epochs):
    # c=0
    pred_actions_epoch, gt_actions_epoch = [], []
    train_loss = 0
    for step in train_data_loader:
        # for overfitting purpose 
        # if c==2:
        #     break
        # c+=1
        coeffs = torch.tensor(make_coeff(num_heads=n_heads, batch_size=step['action'].shape[0])).to(device)
        gt_actions_epoch.append(step['action']) 
        print('the action is:', step['action'].shape , step['action'],step['action'].dtype )
        # coefficients are the alfa values in the paper 
       
        est_q, greedy_idx, greedy_action = model(step['obs'].to(device), step['action'].to(device), coeffs) # inputs: state, act, random_coeff , returns: est_q, greedy_idx, greedy_action
        
        # pred_actions_epoch.append(pred_action)
        # loss = criterion(pred_action.squeeze(), step['action'].squeeze())
        loss= criterion(est_q, greedy_action, step['reward'].to(device), step['terminal'].to(device))  #online_est_q, tar_gd_action, batch_reward, batch_done
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    wandb.log({'train_loss':train_loss/len(train_data_loader)})
    