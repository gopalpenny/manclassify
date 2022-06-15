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
# import re
# from plotnine import *
# import leafmap


# gdrive_path = '/Users/gopal/Google Drive'
# gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# # sys.path.append(gdrive_ml_path)
# out_folder = 'region1'

default_app_path = '/Users/gopal/Google Drive/_Research projects/ML/manclassify/app_data'

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

data_path_files = pd.DataFrame({'Files': os.listdir(st.session_state.proj_path)})


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