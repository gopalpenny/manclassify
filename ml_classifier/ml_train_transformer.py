#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:18:30 2023

@author: gopal
"""

# %%
import torch
import os
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# %%
proj_paths = ["/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data/Thailand",
              "/Users/gopalpenny/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/My Drive/_Research/Research projects/ML/manclassify/app_data/Thailand"]

proj_path = [path for path in proj_paths if os.path.exists(path)][0]


s1 = torch.load(os.path.join(proj_path, 'model_data_s1.pt'))
s2 = torch.load(os.path.join(proj_path, 'model_data_s2.pt'))
classes = torch.load(os.path.join(proj_path, 'model_data_classes.pt'))
classes
# %%
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
        
    return ary




one_hot_encoder = OneHotEncoder(sparse = False)
one_hot_encoder.fit(classes[:,1:])
classes_one_hot = torch.concat((classes[:,0:1], 
                                torch.tensor(one_hot_encoder.transform(classes[:,1:]))), dim = 1)
classes_one_hot

# %%
y_train, y_eval = train_test_split(classes_one_hot, train_size = 0.8, stratify = classes[:, 1])
y_valid, y_test = train_test_split(y_eval, train_size = 0.5, stratify = y_eval[:, 1])

print('y_train count [single, double, plantation, other]:')
print([int(torch.sum(y_train[:,x + 1]).item()) for x in np.arange(classes_one_hot.shape[1]-1)])

print('y_train count [single, double, plantation, other]:')
print([int(torch.sum(y_valid[:,x + 1]).item()) for x in np.arange(classes_one_hot.shape[1]-1)])

print('y_train count [single, double, plantation, other]:')
print([int(torch.sum(y_test[:,x + 1]).item()) for x in np.arange(classes_one_hot.shape[1]-1)])

# %%

y_test

# %%
# get y_train values from loc_train
y_train_df = (loc_train.assign(val = 1) \
  .pivot_table(columns = 'class', index = ['loc_id'], values = 'val', fill_value= 0) \
  .reset_index(['loc_id']))
y_train = y_train_df.to_numpy()
print('y_train:\n',y_train)

# get x_train values from loc_ts_norm (based on loc_id)
x_train = loc_ts_norm[torch.isin(loc_ts_norm[:,0],torch.tensor(y_train[:,0]).to(torch.float64)),:]

# get y_test values from loc_test
y_valid_df = (loc_valid.assign(val = 1) \
  .pivot_table(columns = 'class', index = ['loc_id'], values = 'val', fill_value= 0) \
  .reset_index(['loc_id']))
y_valid = y_valid_df.to_numpy()
print('y_valid:\n',y_valid[0:10,])

# get x_train values from loc_ts_norm (based on loc_id)
x_valid = loc_ts_norm[torch.isin(loc_ts_norm[:,0],torch.tensor(y_valid[:,0]).to(torch.float64)),:]

# get y_test values from loc_test
y_test_df = (loc_test.assign(val = 1) \
  .pivot_table(columns = 'class', index = ['loc_id'], values = 'val', fill_value= 0) \
  .reset_index(['loc_id']))
y_test = y_test_df.to_numpy()
print('y_test:\n',y_test[0:10,])

# get x_train values from loc_ts_norm (based on loc_id)
x_test = loc_ts_norm[torch.isin(loc_ts_norm[:,0],torch.tensor(y_test[:,0]).to(torch.float64)),:]

x_test