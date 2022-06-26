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
st.set_page_config(layout="wide")
import pandas as pd
# import numpy as np
import os
import plotnine as p9
from datetime import datetime
import appmodules.manclass as mf
# import re
# from plotnine import *
# import leafmap

import importlib
importlib.reload(mf)

#%%


# gdrive_path = '/Users/gopal/Google Drive'
# gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# # sys.path.append(gdrive_ml_path)
# out_folder = 'region1'

default_app_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data'

if 'app_path' not in st.session_state:
    st.session_state['app_path'] = default_app_path
    
if 'proj_name' not in st.session_state:
    st.session_state['proj_name'] = 'region1'
    
if 'proj_path' not in st.session_state:
    st.session_state['proj_path'] = os.path.join(st.session_state.app_path, st.session_state.proj_name)
    
    
if 'region_status' not in st.session_state:
    st.session_state['region_status'] = "Not done"
    
if 'samples_status' not in st.session_state:
    st.session_state['samples_status'] = "Not done"
# if 'proj_path' not in st.session_state:
#     st.session_state.proj_path = os.path.join(st.session_state.app_path, st.session_state.proj)

if 'proj_vars' not in st.session_state:
    st.session_state['proj_vars'] = mf.readProjectVars(st.session_state['proj_path'])
    
if 'map_theme' not in st.session_state:
    st.session_state['map_theme'] = p9.theme(panel_background = p9.element_rect(fill = None),      
                     panel_border = p9.element_rect(),
                     panel_grid_major=p9.element_blank(),
                     panel_grid_minor=p9.element_blank(),
                     plot_background=p9.element_rect(fill = None))
    

if st.checkbox("Display Session Variables"):
    st.markdown('* st.session_state.app_path: ' + st.session_state.app_path +
                '\n* st.session_state.proj_name: ' + st.session_state.proj_name +
                '\n* st.session_state.proj_path: ' + st.session_state.proj_path)

st.session_state.app_path = st.text_input("Application directory (in Google Drive)", value = default_app_path)

projects = os.listdir(st.session_state.app_path)

# %%
# os.walk(default_app_path)
# %%

st.title("Dashboard")

st.session_state.proj_name = st.selectbox("Select a project", options = tuple(projects))
st.session_state.proj_path = os.path.join(st.session_state.app_path, st.session_state.proj_name)


start_date_str = (st.session_state['proj_vars']['classification_start_month'] + ' ' +
                  str(st.session_state['proj_vars']['classification_start_day']) + ', ' +
                  str(st.session_state['proj_vars']['classification_year_default']))


st.markdown("""### Project timespan
            
Timeseries will be downloaded for dates contained within this timespan.
Pixels can only be classified for years within this timespan. If the end
date is less than six months into the year, that year will be excluded.
Be careful changing this timespan if you've already started downloading timeseries
data.
            """)

m0cols = st.columns([1,1,1])
with m0cols[0]:
    prior_start_date = datetime.strptime(st.session_state['proj_vars']['proj_start_date'] , '%Y-%m-%d')
    project_start_datetime = st.date_input('Start date', 
                                           value = prior_start_date)
    project_start_date = datetime.strftime(project_start_datetime, "%Y-%m-%d")
with m0cols[1]:
    prior_end_date = datetime.strptime(st.session_state['proj_vars']['proj_end_date'], '%Y-%m-%d')
    project_end_datetime = st.date_input('End date',
                                         value = prior_end_date)
    project_end_date = datetime.strftime(project_end_datetime, "%Y-%m-%d")
with m0cols[2]:
    st.markdown('#')
    st.button('Set project timespan (NOT WORKING)',
               on_click = mf.setProjectTimespan, 
               args = (project_start_date, project_end_date, st.session_state['proj_path'], ))

# st.number_input()

st.markdown('### Start date for classification year (current: )', )
m1cols = st.columns([1, 1, 1, 2])


with m1cols[0]:
    proj_years = st.session_state['proj_vars']['proj_years']
    default_year = st.session_state['proj_vars']['classification_year_default']
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
              on_click = mf.setProjectStartDate, 
              args = (year_default, start_month, start_day, st.session_state['proj_path'], ))
    
data_path_files = pd.DataFrame({'Files': os.listdir(st.session_state.proj_path)})


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
        # st.write(data_path_files)
        # dirlist = []
        
        # for dir in os.listdir(st.session_state.proj_path):
        #     for file in os.listdir(os.path.join(st.session_state.proj_path, dir)):
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