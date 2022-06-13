#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:42:12 2022

@author: gopal
"""



import streamlit as st
import pandas as pd
# import numpy as np
import geopandas as gpd
# import geemap
import os
import ee
import sys
# import plotnine as p9
# import re
import importlib
import appmodules.manclass as mf

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)
# from geemod import rs
# from geemod import eesentinel as ees



# region_path = os.path.join(st.session_state.proj_path,"region")
# region_shp_path = os.path.join(region_path,"region.shp")
# sample_locations_path = os.path.join(st.session_state.proj_path,st.session_state.proj_name + "_sample_locations/sample_locations.shp")

app_path = '/Users/gopal/Google Drive/_Research projects/ML/manclassify/app_data'
proj_path = os.path.join(app_path, 'region1')
region_shp_path = os.path.join(proj_path,'region','region.shp')

proj_name = 'region1'
sample_locations_path = os.path.join(app_path, proj_name,proj_name + "_sample_locations/sample_locations.shp")

region_status = os.path.exists(region_shp_path)
sample_status = os.path.exists(sample_locations_path)


# %%
timeseries_dir_name = proj_name + "_pt_timeseries"
timeseries_dir_path = os.path.join(proj_path, timeseries_dir_name)
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
# 
importlib.reload(mf)
#
loc = gpd.read_file(sample_locations_path)

ts_status_path = mf.TimeseriesStatusInit(proj_path)
ts_status = pd.read_csv(ts_status_path)



# pt_iloc_list = 0:5
date_range = ['2014-01-01', '2022-04-30']

# %%

# Number of points to download at a time
NumPts = int(st.number_input('Number of points', 1, value = 5))

# Points not yet downloaded
loc_notrun = ts_status.loc[~ts_status.allcomplete].loc_id
loc_selected = loc.loc[loc.loc_id.isin(loc_notrun)].iloc[0:NumPts]

# def five():
#     return 5
# test = 0
# test = st.button('Test', on_click = five, args = ())
# st.write('test = ' + str(test))
st.button('Download points', on_click = mf.DownloadPoints, args = (loc_selected, date_range, timeseries_dir_path,ts_status,))

# st.write(ts_status[0:10])
st.text('Points to download')



if st.checkbox("Display Sample location data"):
    st.write(loc_selected)
    st.write(ts_status)

