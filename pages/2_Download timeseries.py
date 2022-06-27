#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:42:12 2022

@author: gopal
"""



import streamlit as st
st.set_page_config(page_title="Grab sample data", layout="wide", page_icon="🌏")
import pandas as pd
# import numpy as np
import geopandas as gpd
# import geemap
import os
import ee
import sys
import plotnine as p9
# import re
import importlib
import appmodules.manclass as mf

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)
# from geemod import rs
# from geemod import eesentinel as ees


app_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data'
proj_path = os.path.join(app_path, 'region1')

sample_locations_dir_path = os.path.join(st.session_state.proj_path,st.session_state.proj_name + "_sample_locations")
sample_locations_path = os.path.join(sample_locations_dir_path, "sample_locations.shp")
region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")

proj_name = 'region1'

region_status = os.path.exists(region_shp_path)
sample_status = os.path.exists(sample_locations_path)

# %%
timeseries_dir_name = proj_name + "_download_timeseries"
timeseries_dir_path = os.path.join(proj_path, timeseries_dir_name) 


# %%

# 
importlib.reload(mf)
#
loc = gpd.read_file(sample_locations_path)

ts_status_path = mf.TimeseriesStatusInit(proj_path)

# def plot(region_status, sample_status, ts_status_path, )
ts_status = pd.read_csv(ts_status_path)
ts_status_downloaded = ts_status.loc[ts_status.allcomplete].loc_id
loc_downloaded = loc.loc[loc.loc_id.isin(ts_status_downloaded)]

# Points not yet downloaded
ts_status_notdownloaded = ts_status.loc[~ts_status.allcomplete].loc_id
loc_notdownloaded = loc.loc[loc.loc_id.isin(ts_status_notdownloaded)]

# %%

if not region_status:
    p_map = p9.ggplot() + p9.geom_blank() + p9.coord_equal()
else:
    region_shp = gpd.read_file(region_shp_path)
    p_map = (p9.ggplot() + st.session_state.map_theme +
             p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black"))
    
    if sample_status:
        # sample_locations_shp = gpd.read_file(sample_locations_path)
        p_map = (p_map + 
                 p9.geom_map(data = loc_notdownloaded, mapping = p9.aes(), color = 'black', shape = '+', size = 1) + 
                 p9.geom_map(data = loc_downloaded, mapping = p9.aes(), shape = 'o', fill = 'red', color = None, size = 2))




col1, col2, col3 = st.columns([1,2,1])

# with col1:
#     st.pyplot(p9.ggplot.draw(p_map))
with col2:
    st.pyplot(p9.ggplot.draw(p_map))

# if not os.path.exists(timeseries_dir_path): os.mkdir(timeseries_dir_path)

    


# %%
# loc['lon'] = loc.geometry.x
# loc['lat'] = loc.geometry.y

# for i in [1, 10, 100]:
#     print(i)


# %%
        
if 'ee_initialized' not in st.session_state:
    st.session_state['ee_initialized'] = False
    
if not st.session_state.ee_initialized:
    ee.Initialize()
    st.session_state.ee_initialized = True



# pt_iloc_list = 0:5
try:
    date_range = ([st.session_state['proj_vars']['proj_start_date'], 
                   st.session_state['proj_vars']['proj_end_date']])
except:
    st.error('Need to set project timespan before proceeding')

# %%

# Number of points to download at a time
NumPts = int(st.number_input('Number of points', 1, value = 5))

# points to download
loc_selected_numpts = loc_notdownloaded.iloc[0:NumPts]

# def five():
#     return 5
# test = 0
# test = st.button('Test', on_click = five, args = ())
# st.write('test = ' + str(test))
st.markdown("Date range to download date: `" + date_range[0] + '` to `' + date_range[1] + '`')
st.button('Download points', on_click = mf.DownloadPoints, args = (loc_selected_numpts, date_range, timeseries_dir_path,ts_status,))

# st.write(ts_status[0:10])
st.text('Points to download')



if st.checkbox("Display Sample location data"):
    st.write(loc_selected_numpts)
    st.write(ts_status)

