#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 21:35:24 2023

@author: gopal
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

class s2Dataset(Dataset):
    """Sentinel 2 dataset"""
    
    def __init__(self, x, y, max_obs):
        """
        Args:
            x (tensor): contains loc_id and predictors as columns, s2 observations as rows
            y (tensor): contains loc_id as rows (& first column), class as 1-hot columns
        """
        self.x = x
        self.y = y
        self.max_obs = max_obs
        # self.proj_path = proj_path
        # proj_normpath = os.path.normpath(proj_path)
        # proj_dirname = proj_normpath.split(os.sep)[-1]
        # self.proj_name = re.sub("_classification$","",proj_dirname)
        # self.class_path = os.path.join(proj_path, self.proj_name + "_classification")
        # self.ts_path = os.path.join(proj_path, self.proj_name + "_download_timeseries")
        # self.pt_classes = pd.read_csv(os.path.join(self.class_path,"location_classification.csv"))
        # self.pt_classes = classes[['loc_id', class_colname]].dropna()
        # self.classes = pd.unique(self.pt_classes[class_colname])
        # self.labels = self.pt_classes.assign(val = 1).pivot_table(columns = class_colname, index = 'loc_id', values = 'val', fill_value= 0)

    
    def __getitem__(self, idx):
        # get loc_id
        loc_id = self.y[idx,0]
        self.last_loc_id = loc_id
        
        # select location id
        x_loc = self.x[self.x[:,0]==loc_id]
        x_prep = x_loc[:,1:] # remove loc_id column
        
        # pad zeros to max_obs
        n_pad = self.max_obs - x_prep.shape[0]
        
        x = torch.cat((x_prep, torch.zeros(n_pad, x_prep.shape[1])), dim = 0)
        
        x = x.float()
        
        
        
        # get one-hot encoding for the point as tensor
        y = torch.tensor(self.y[idx,1:]).float()
        
        return x, y
        
    def __len__(self):
        return self.y.shape[0]