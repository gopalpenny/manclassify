#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:03:28 2022

@author: gopal
"""


import streamlit as st
st.set_page_config(page_title="Sample region", layout="wide", page_icon="🌍")
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

# %%
# region_path = os.path.join(st.session_state.proj_path,"region")
sample_locations_dir_path = os.path.join(st.session_state.proj_path,st.session_state.proj_name + "_sample_locations")
sample_locations_path = os.path.join(sample_locations_dir_path, "sample_locations.shp")
region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")

region_status = os.path.exists(region_shp_path)
sample_status = os.path.exists(sample_locations_path)

if not region_status:
    p_map = p9.ggplot() + p9.geom_blank()
else:
    region_shp = gpd.read_file(region_shp_path)
    p_map = (p9.ggplot() + st.session_state.map_theme +
             p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black"))
    
    if sample_status:
        sample_locations_shp = gpd.read_file(sample_locations_path)
        p_map = (p_map + 
                 p9.geom_map(data = sample_locations_shp, mapping = p9.aes(color = 'loc_id')))


# %% Layout

st.title("Sampling region")

col1, col2, col3 = st.columns([1,2,1])

# with col1:
#     st.pyplot(p9.ggplot.draw(p_map))
with col2:
    st.pyplot(p9.ggplot.draw(p_map))

# %%


path_to_shp_import = st.text_input('Path to shapefile',
              value = '/Users/gopal/Projects/ArkavathyTanksProject/arkavathytanks/spatial/CauveryBasin/Cauvery_boundary5.shp')
st.button('Import shapefile', on_click = mf.ImportShapefile, args = (sample_locations_dir_path, path_to_shp_import,))
if region_status: st.text('Already done (' + re.sub(st.session_state.app_path,'',region_shp_path) + ')')
st.button('Generate sample locations', on_click = mf.GenerateSamples, args = (st.session_state.app_path,st.session_state.proj_name,))
if sample_status: st.text('Already done (' + re.sub(st.session_state.app_path,'',sample_locations_path) + ')')

# gpd.read_file()

# foo = pd.DataFrame({'a':[1, 2, 3]})

# st.write(st.session_state.proj_path)


# def GenerateSamplePixels

# %%

if st.checkbox("Display Session Variables"):
    st.markdown('* st.session_state.app_path: ' + st.session_state.app_path +
                '\n* st.session_state.proj_name: ' + st.session_state.proj_name +
                '\n* st.session_state.proj_path: ' + st.session_state.proj_path)
    