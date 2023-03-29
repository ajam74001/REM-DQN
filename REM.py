import numpy as np
# import tensorflow as tf
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import wandb
from torchmetrics.classification import Accuracy, Precision, Recall
## from .dqn import DQN
class DQN_REM(nn.Module):
    def __init__(self,input_dim=25,fc1_dim=256, fc2_dim=256, num_actions=5,num_heads=200):
        super().__init__()
        self.num_actions= num_actions
        self.num_heads = num_heads
        self.MLP= nn.Sequential(nn.Linear(input_dim, fc1_dim), nn.ReLU(), nn.Linear(fc1_dim, fc2_dim), nn.ReLU())
        self.fc = nn.Linear(fc2_dim, num_actions*num_heads)

    def forward(self,state, act, random_coeff):
        # random_coeff :  Bx1xN_heads
        # state B x25
        # act: Bx 1

        x= self.MLP(state)
        x= self.fc(x)
        # continue with reshaping and greedy actions 
        out = x.view(-1, self.num_actions, self.num_heads) # out : B x n_actions x  n_heads

        out_rem = out * random_coeff  #B x n_actions x  n_heads
        out_mean= out.mean(dim=2) #B x n_actions
        # print('the out mean', out_mean)
        greedy_idx = torch.argmax(out_mean, dim=1) # B
        print(greedy_idx)
        action_mask = nn.functional.one_hot(act.squeeze(1), self.num_actions).float().unsqueeze(2) # B x n_action x 1
        greedy_action_mask = nn.functional.one_hot(greedy_idx, self.num_actions).float().unsqueeze(2) # B x n_action x 1
        # (out_rem * action_mask) -> # B x n_actions x  n_heads
        est_q = (out_rem * action_mask).sum(dim=1).sum(dim=1) # B
        greedy_action = (out_rem * greedy_action_mask).sum(dim=1).sum(dim=1)  # B

        return est_q, greedy_idx, greedy_action
         

