#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:40:03 2022

@author: gopal
"""

import streamlit as st
import pandas as pd
# import numpy as np
import os
import re
from plotnine import *
import leafmap


gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# sys.path.append(gdrive_ml_path)
out_folder = 'gee_sentinel_ts'
data_path = os.path.join(gdrive_ml_path, 'manclassify/script_output', out_folder)

# %%

st.set_page_config(page_title="Classify", page_icon="🌍")



data_path = st.text_input("Path to Google Drive folder", value = data_path)

data_path_files = pd.DataFrame({'Files': os.listdir(data_path)})


# files = os.listdir(data_path)

# %%
col1, col2 = st.columns(2)


i = 0

time_series_pd_load = pd.read_csv(os.path.join(data_path,data_path_files.Files.iloc[i]))

# %%
time_series_pd = time_series_pd_load
time_series_pd['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in time_series_pd_load['image_id']]
time_series_pd['datetime'] = pd.to_datetime(time_series_pd['datestr'])
time_series_pd = time_series_pd[['datetime','B8','B4','B3','B2','cloudmask']]
time_series_pd = time_series_pd.assign(NDVI = lambda df: (df.B8 - df.B4)/(df.B8 + df.B4))
# time_series_pd.assign(NDVI = ['NDVI'] = (time_series_pd['B8 - time_series_pd.B4) / (time_series_pd.B8 + time_series_pd.B4)
time_series_long = time_series_pd.melt(id_vars = ['datetime', 'cloudmask'], value_vars = ['B8','B4','B3','B2','NDVI'])
# time_series_long


# %%


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

with st.sidebar:
    lat = st.number_input('Lat', 0.0, 90.0, 13.0) #, step = 0.1)
    lon = st.number_input('Lon', -180.0, 180.0, 77.0) #, step = 0.1)
    # st.write(data_path_files)
    if st.checkbox('Show project files'):
        st.text('Files in ' + data_path)
        st.write(data_path_files)
    
# with col1:
m = leafmap.Map(center=(lat, lon), zoom=18)
m.add_tile_layer(
    url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    name="Google Satellite",
    attribution="Google",
)


# with col1:
#     st.text("Col 1")
    
# with col2:
st.pyplot(ggplot.draw(p_nrgb))
st.pyplot(ggplot.draw(p_NDVI))

st.write(m.to_streamlit())
