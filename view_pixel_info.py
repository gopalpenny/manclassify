#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:40:03 2022

@author: gopal
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from plotnine import *

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# sys.path.append(gdrive_ml_path)
out_folder = 'gee_sentinel_ts'
data_path = os.path.join(gdrive_ml_path, 'manclassify/script_output', out_folder)

# %%

files = os.listdir(data_path)

# %%

i = 0

time_series_pd_load = pd.read_csv(os.path.join(data_path,files[i]))

# %%
time_series_pd = time_series_pd_load
time_series_pd['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in time_series_pd_load['image_id']]
time_series_pd['datetime'] = pd.to_datetime(time_series_pd['datestr'])
time_series_pd = time_series_pd[['datetime','B8','B4','B3','B2','cloudmask']]
time_series_pd = time_series_pd.assign(NDVI = lambda df: (df.B8 - df.B4)/(df.B8 + df.B4))
# time_series_pd.assign(NDVI = ['NDVI'] = (time_series_pd['B8 - time_series_pd.B4) / (time_series_pd.B8 + time_series_pd.B4)
time_series_long = time_series_pd.melt(id_vars = ['datetime', 'cloudmask'], value_vars = ['B8','B4','B3','B2','NDVI'])
# time_series_long


# In[35]:


p_nrgb = (ggplot(data = time_series_long.query('cloudmask == 0 & variable != "NDVI"')) + 
   geom_point(aes(x = 'datetime', y = 'value', color = 'variable')) +
   scale_x_datetime(date_labels = '%Y-%b') +
   theme(figure_size = (10,5)))
p_NDVI = (ggplot(data = time_series_long.query('cloudmask == 0 & variable == "NDVI"')) + 
   geom_point(aes(x = 'datetime', y = 'value', color = 'variable')) +
   scale_x_datetime(date_labels = '%Y-%b') +
   theme(figure_size = (10,5)))



# %%

st.title("Pixel classification")

# In[36]:


st.pyplot(ggplot.draw(p_nrgb))



# %% 

st.pyplot(ggplot.draw(p_NDVI))
