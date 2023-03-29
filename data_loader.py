# dataloader 
import numpy as np
import pandas as pd
# import torch 
# import torchvision 
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import math
# from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import scipy.io
import scipy.io as sio
from torch.utils.data import Dataset
import torch

class Dataloader(Dataset):
    def __init__(self, path_list):
        self.obs = np.array(pd.read_csv(path_list[0])).astype(np.float32)
        self.action = np.array(pd.read_csv(path_list[1]))
        self.reward = np.array(pd.read_csv(path_list[2])).astype(np.float32)
        self.terminal = np.array(pd.read_csv(path_list[3])).astype(np.float32)

    def __len__(self):
        return len(self.terminal)

    def __getitem__(self, index):
        ret={}
        ret['obs'] = self.obs[index,:]
        ret['reward'] = self.reward[index] 
        ret['action'] = self.action[index]
        ret['terminal'] = self.terminal[index]
        return ret
