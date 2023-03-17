import torch 
import torch.nn as nn
import torch.nn.functional as F
# est_q, greedy_idx, greedy_action
class loss_fn(nn.Module):
    def __init__(self,gamma,  delta=1.0):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        
    def forward(self,online_est_q, tar_gd_action, batch_reward, batch_done):
            max_q_target = batch_reward + self.gamma * (1 - batch_done) * tar_gd_action 
            loss = F.huber_loss(max_q_target, online_est_q, reduction = 'mean', delta = self.delta)
            return loss



