#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:42:12 2022

@author: gopal
"""



import streamlit as st
import pandas as pd
import geopandas as gpd
import geemap
import os
import ee
import sys
import plotnine as p9
import re
import appmodules.manclass as mf

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemod import rs
from geemod import eesentinel as ees



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

loc = gpd.read_file(sample_locations_path)

# %%
timeseries_dir_name = proj_name + "_pt_timeseries"
timeseries_dir_path = os.path.join(proj_path, timeseries_dir_name)
if not os.path.exists(timeseries_dir_path): os.mkdir(timeseries_dir_path)

# %%
# loc['lon'] = loc.geometry.x
# loc['lat'] = loc.geometry.y

ee.Initialize()

i = 1
pt_gpd = loc.iloc[i]
sample_pt_coords = [pt_gpd.geometry.x, pt_gpd.geometry.y]

date_range = ['2014-01-01', '2022-04-30']

sample_pt_id = loc.loc_id.iloc[i]
sample_pt_name = 'pt_ts_loc' + str(sample_pt_id)

DownloadSamplePt(sample_pt_coords, sample_pt_name, timeseries_dir_path, date_range)


