#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:54:31 2022

@author: gopal
"""

# This is the app dashboard, the home base for classifying images

# The subpages will include:
# Generate sample
# Generate data
# Classify
    

# The app dashboard will include
# 1. The Google Drive main directory
# 2. Map of current data
#   a. If Shapefile is present, include
#   b. If sample points are present, include them
#   c. If sample points have data, use a different shape

import streamlit as st
st.set_page_config(page_title="Overview", layout="wide", page_icon="🌏")
import pandas as pd
import numpy as np
import os
import plotnine as p9
from datetime import datetime
import appmodules.manclass as mf
# import appmodules.SamplePageFunctions as spf
import appmodules.OverviewPageFunctions as opf
# import re
# from plotnine import *
# import leafmap

import importlib
importlib.reload(mf)
importlib.reload(opf)



#%%


# gdrive_path = '/Users/gopal/Google Drive'
# gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# # sys.path.append(gdrive_ml_path)
# out_folder = 'region1'

default_app_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data'
    

st.title("Dashboard")


if 'app_path' not in st.session_state:
    st.session_state['app_path'] = default_app_path

st.markdown('`app_path`')
st.write(st.session_state['app_path'])


project_columns = st.columns([3,1])

with project_columns[0]:
    st.text_input("Application directory (must be in Google Drive and synced locally)", on_change = opf.UpdateAppPath,
                  value = default_app_path, key = 'app_path_box')
    
if 'proj_name' not in st.session_state:
    projects_all_init = os.listdir(st.session_state.app_path)
    st.session_state['proj_name'] = projects_all_init[0]


opf.checkProjStatus()

with project_columns[1]:
    projects_all = os.listdir(st.session_state.app_path)
    projects = [f for f in projects_all if not f.startswith('.')] + ['Create new project']
    st.selectbox("Select a project", options = tuple(projects), 
                  key = 'proj_name_box',
                  on_change = opf.UpdateProjName)
    if st.session_state['proj_name_box'] == 'Create new project':
        st.text_input('New project name',key = 'new_project_name', on_change = opf.CreateNewProject)


start_date_str = (st.session_state['proj_vars']['classification_start_month'] + ' ' +
                  str(st.session_state['proj_vars']['classification_start_day']) + ', ' +
                  str(st.session_state['proj_vars']['classification_year_default']))


with st.expander('View project directory structure'):
    st.markdown("""
The `project_path` is determined as `app_path/project_name`. The structure of this directory is:
    
    ```
    project_path
    - project_vars.json
    - region1_classification/
      - location_classification.csv
    - region1_download_timeseries/
      - pt_ts_loc0_s2.csv
      - pt_ts_loc0_s1.csv
    - region1_sample_locations/
      - region.shp
      - random_locations.shp
      - sample_locations.shp
    ```
                """)

st.markdown("""### Project timespan
            
Timeseries will be downloaded for dates contained within this timespan.
Pixels can only be classified for years within this timespan.
Be careful changing this timespan if you've already started downloading timeseries
data.
            """)

m0cols = st.columns([1,1,1])
with m0cols[0]:
    prior_start_date = datetime.strptime(st.session_state['proj_vars']['proj_start_date'] , '%Y-%m-%d')
    project_start_datetime = st.date_input('Start date (' + st.session_state['proj_vars']['proj_start_date'] + ')', 
                                           value = prior_start_date, key = 'overview_proj_start_date')
    project_start_date = datetime.strftime(project_start_datetime, "%Y-%m-%d")
with m0cols[1]:
    prior_end_date = datetime.strptime(st.session_state['proj_vars']['proj_end_date'], '%Y-%m-%d')
    project_end_datetime = st.date_input('End date (' + st.session_state['proj_vars']['proj_end_date'] + ')',
                                         value = prior_end_date)
    project_end_date = datetime.strftime(project_end_datetime, "%Y-%m-%d")
with m0cols[2]:
    st.markdown('#')
    st.button('Set project timespan',
               on_click = opf.setProjectTimespan, 
               args = (project_start_date, project_end_date, ))

# st.number_input()

class_start_date_format = (st.session_state['proj_vars']['classification_start_month'] + 
                           ' ' + str(st.session_state['proj_vars']['classification_start_day']) + ', ' +
                           str(st.session_state['proj_vars']['classification_year_default']))
st.markdown('### Start date for classification year (current: ' + class_start_date_format + ')', )
st.markdown("""
`Default year` is the default year for classification (the Classify page will initialize to this year).
`Month` and `Day` represent the start date
for the classification for each year.
            """)
m1cols = st.columns([1, 1, 1, 2])



with m1cols[0]:
    proj_years = st.session_state['proj_vars']['proj_years']
    default_year = st.session_state['proj_vars']['classification_year_default']
    if default_year < np.min(proj_years):
        default_year = int(np.min(proj_years))
        st.session_state['proj_vars']['classification_year_default'] = default_year
    elif default_year > np.max(proj_years):
        default_year = int(np.max(proj_years))
        st.session_state['proj_vars']['classification_year_default'] = default_year
    idx_default_year = [i for i in range(len(proj_years)) if proj_years[i] == default_year][0]
    year_default = st.selectbox("Default year", options = proj_years, 
                                    index = idx_default_year)
    
with m1cols[1]:
    year_start_month = [datetime.strftime(datetime.strptime('2000-' + str(x) + '-01','%Y-%m-%d'), '%B') for x in range(1,13)]
    current_month = st.session_state['proj_vars']['classification_start_month']
    idx_month = [i for i in range(len(year_start_month)) if year_start_month[i] == current_month][0]
    start_month = st.selectbox("Month", options = year_start_month, index = idx_month)
with m1cols[2]:
    start_day = st.number_input("Day", min_value = 1, max_value = 31, value = st.session_state['proj_vars']['classification_start_day'])
# with m1[2]:
#     st.markdown('####')
#     st.markdown('#### Start day for classification year')
with m1cols[3]:
    st.markdown('#')
    st.button('Set start date for classification year', 
              on_click = opf.setClassificationStartDate, 
              args = (year_default, start_month, start_day, ))


st.write(st.session_state['proj_vars'])


st.markdown('### App overview')

st.markdown(
    """
    This app is used to generate sample points and classify them based on timeseries
    data from Google Earth engine. The key steps for the app are:
    
    1. Create a Google Drive directory with a polygon shapefile for sampling
    2. Select the shapefile and generate sample points
    3. Generate timeseries data from those sample points
    4. Classify each point one by one
        
    The result will produce a .csv file containing the resulting classification
    for each point.
    
    Data is stored in the project subdirectories. This includes:
        
        * /region/region.shp : shapefile for the region
        * /stuff
    """)
    
    
# %%


os.listdir(default_app_path)[0:4]

# %%


from streamlit_folium import st_folium
import folium

if 'counter' not in st.session_state:
    st.session_state.counter = 1

m = folium.Map()
tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
        ).add_to(m)
# tile1 = folium.TileLayer(
#         tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
#         attr = 'Google',
#         name = 'Google Satellite',
#         overlay = False,
#         control = True
#        ).add_to(m)





with st.sidebar:
    st.number_input('num', 0, 5, 2)
    
# col1, col2 = st.columns(2)

st_folium(m, height = 300, width = 600)

with st.sidebar:
    st.subheader("Project: " + st.session_state.proj_name)
    if st.checkbox('Show project files'):
        st.text('Files in ' + st.session_state.proj_path)
        for directory in os.listdir(st.session_state.proj_path):
            dirpath = os.path.join(st.session_state.proj_path, directory)
            if os.path.isdir(dirpath):
                st.write(directory + ":")
                for file in os.listdir(dirpath):
                    st.write("- " + file)
            else:
                st.write("dir " + directory)
                
        # filelist=[]
        # for root, dirs, files in os.walk(st.session_state.proj_path):
        #       for file in files:
        #              filename=os.path.join(file)
        #              filelist.append(filename)
        # st.write(filelist)