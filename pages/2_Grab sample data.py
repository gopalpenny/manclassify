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

loc = gpd.read_file(sample_locations_path)

ts_status_path = mf.TimeseriesStatusInit(proj_path)
ts_status = pd.read_csv(ts_status_path)



# pt_iloc_list = 0:5
date_range = ['2014-01-01', '2022-04-30']
# timeseries_dir_path
# date_range

# DownloadPoints(loc, pt_iloc_list, date_range, timeseries_dir_path)

# NumPts = 5
# loc_download = loc.iloc[1:3]

# %%

# s1_colname = 'pt_ts_loc_s1'
# mf.TimeseriesUpdateStatus(loc_id, s1_colname, 'Running', timeseries_dir_path)

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

st.text('Points to download')
st.write(loc_selected)


st.write(ts_status)
# # if region_status: st.text('Already done (' + re.sub(st.session_state.app_path,'',region_shp_path) + ')')

# def PrepDownloadPoints(loc, date_range, timeseries_dir_path):
#     loc_selected
#     DownloadPoints(loc_selected, date_range, timeseries_dir_path)

# %%
# loc_id = 4
# colname = 's1'
# new_status = np.nan

# %%
# rowList = [np.nan]
# type(str(rowList[1]))

# st.button()
    

# %%
    
    

# def TimeseriesStatusInit(timeseries_dir_path):
    
# """
# Operations: 
# 1. update status manually (force status update) - after starting EE task
# -- a. add status for file if column does not exist
# -- b. change status for file if it does exist
# 2. update status automatically (check files, check earth engine)
# -- a. look for all possible combinations of files
# -- b. add status for file if it does not exist
# -- c. change status for file if it does exist
# 2. check status
# -- a. status present in file
# -- b. status not present in file

# eeGetTimeseriesStatus
# """
    

# %%

# def GetStatus(timeseries_dir_path):
#     if not os.path.exists(timeseries_dir_path): os.mkdir(timeseries_dir_path)
#     ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    
#     if not os.path.exists(timeseries_dir_path):

# def CheckLocFiles(loc_id, ts_all_filenames):
#     loc_filenames = [re.sub('loc_','loc' + loc_id + '_', x) for x in ts_all_filenames]
#     return loc_filenames
    
# def CheckLocTimeseries(loc_id, timeseries_dir_path):
#     loc_id
    
#     ts_all_files = os.listdir(timeseries_dir_path)
#     ts_all_filenames = list(set([re.sub('loc[0-9]+_','loc_',x) for x in ts_all_files]))
#     ts_all_filenames
    
#     for 
# %%
# for i in range(loc_download.shape[0]):
#     print(i)
#     timeseries_dir_path
#     s1_pt_filename = sample_pt_name + '_s1'
#     s1_pt_filepath = os.path.join(timeseries_dir_path, s1_pt_filename + '.csv')
#     s2_pt_filename = sample_pt_name + '_s2'
#     s2_pt_filepath = os.path.join(timeseries_dir_path, s2_pt_filename + '.csv')
    
# %%


# %%

