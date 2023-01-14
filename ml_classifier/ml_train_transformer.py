#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:18:30 2023

@author: gopal
"""

# %%
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from ml_classifier.ml_transformer import SentinelDatasets, TransformerClassifier

# %%
proj_paths = ["/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data/Thailand",
              "/Users/gopalpenny/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/My Drive/_Research/Research projects/ML/manclassify/app_data/Thailand"]

proj_path = [path for path in proj_paths if os.path.exists(path)][0]


s1_all = torch.load(os.path.join(proj_path, 'model_data_s1.pt'))
s2_all = torch.load(os.path.join(proj_path, 'model_data_s2.pt'))
classes = torch.load(os.path.join(proj_path, 'model_data_classes.pt'))
classes

# %%
s1 = None
s2 = None

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

# Split training X values
s1_train = s1_all[torch.isin(s1_all[:, 0], y_train[:,0])]
s1_valid = s1_all[torch.isin(s1_all[:, 0], y_valid[:,0])]
s1_test = s1_all[torch.isin(s1_all[:, 0], y_test[:,0])]

s2_train = s2_all[torch.isin(s2_all[:, 0], y_train[:,0])]
s2_valid = s2_all[torch.isin(s2_all[:, 0], y_valid[:,0])]
s2_test = s2_all[torch.isin(s2_all[:, 0], y_test[:,0])]

# %%

# Prep datasets
data_train = SentinelDatasets(s1_train, s2_train, y_train, 64, 64)
data_valid = SentinelDatasets(s1_valid, s2_valid, y_valid, 64, 64)
data_test = SentinelDatasets(s1_test, s2_test, y_test, 64, 64)

# %%

# Prep weighted sampling for training data

# adapted from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
target_classes = torch.stack([torch.argmax(data_train.__getitem__(i)[2]) for i in range(data_train.__len__())])
# count of samples in each class
class_sample_count = np.array([torch.sum(target_classes == i) for i in torch.unique(target_classes)])

# weight for each class (classes must go from 0 to n-1 classes)
weight = 1. / class_sample_count
sample_weights = np.array([weight[i] for i in target_classes])
sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights))


# %%
# Prep dataloaders

train_dl = DataLoader(data_train, batch_size = 20, drop_last = True, sampler = sampler)
valid_dl = DataLoader(data_valid, batch_size = 20, drop_last = False)
test_dl = DataLoader(data_test, batch_size = 20, drop_last = False)


# %%

xnn = TransformerClassifier(64, dmodel = 36, nhead = 6, dhid = 100, nlayers = 3, data_dim = 5, nclasses = 4)

# %%
# test transformer
s1, s2, y = next(iter(train_dl))
xnn(s2)

# %%

def get_num_correct(model_batch_out, y_batch):
    pred = torch.argmax(model_batch_out, dim = 1)
    actual = torch.argmax(y_batch, dim = 1)
    num_correct = torch.sum(pred == actual).item()
    # print('type',type(num_correct))
    # x = num_correct# item()
    # print('num_correct', num_correct.item())
    return num_correct


# %%

# Training loop

# i = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(xnn.parameters(), lr = 0.001)


# print(i)
# for train_features, train_labels in train_dl:
#     i += 1
#     print(i)
n_epochs = 100
loss_hist_train = [0] * n_epochs
accuracy_hist_train = [0] * n_epochs
loss_hist_valid = [0] * n_epochs
accuracy_hist_valid = [0] * n_epochs
for epoch in range(n_epochs):
    xnn.train()
    print(f"Beginning epoch {epoch+1} / {n_epochs}")
    for _, x_batch, y_batch in train_dl:
        
        # Forward pass
        pred = xnn(x_batch)
        loss = loss_fn(pred, y_batch)
        
        # Accumulate loss and accuracy
        loss_hist_train[epoch] += loss.item() * y_batch.size(0)
        accuracy_hist_train[epoch] += get_num_correct(pred, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loss_hist_train[epoch] /= float(len(train_dl.dataset))
    accuracy_hist_train[epoch] /= float(len(train_dl.dataset))
    
    with torch.no_grad():
        for _, x_batch, y_batch in train_dl:

            # Forward pass
            pred = xnn(x_batch)
            loss = loss_fn(pred, y_batch)

            # Accumulate loss and accuracy
            loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
            accuracy_hist_valid[epoch] += get_num_correct(pred, y_batch)

        loss_hist_valid[epoch] /= float(len(train_dl.dataset))
        accuracy_hist_valid[epoch] /= float(len(train_dl.dataset))
        
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss_hist_train[epoch]:.4f}, Accuracy: {accuracy_hist_train[epoch]:.4f}'
          f' Val Accuracy: {accuracy_hist_valid[epoch]:.4f}')
        

# %%
fig, axs = plt.subplots(2)
axs[0].plot(loss_hist_train)
axs[0].plot(loss_hist_valid)
axs[0].set(ylabel = "Loss")
axs[1].plot(accuracy_hist_train)
axs[1].plot(accuracy_hist_valid)
axs[1].set(ylabel = "Accuracy")