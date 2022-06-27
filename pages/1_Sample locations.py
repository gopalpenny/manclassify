#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:03:28 2022

@author: gopal
"""


import streamlit as st
st.set_page_config(page_title="Sample region", layout="wide", page_icon="🌏")
import pandas as pd
import geopandas as gpd
import geemap
import os
import ee
import sys
import plotnine as p9
import re
import numpy as np
from streamlit_folium import st_folium
import folium
# import appmodules.manclass as mf
import appmodules.SamplePageFunctions as spf

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research/Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemod import rs


import importlib
importlib.reload(spf)

# %%
# region_path = os.path.join(st.session_state.proj_path,"region")
sample_locations_dir_path = os.path.join(st.session_state.proj_path,st.session_state.proj_name + "_sample_locations")
random_locations_path = os.path.join(sample_locations_dir_path, "random_locations.shp")
sample_locations_path = os.path.join(sample_locations_dir_path, "sample_locations.shp")
region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")

region_status = os.path.exists(region_shp_path)
random_status = os.path.exists(random_locations_path)
sample_status = os.path.exists(sample_locations_path)

if not region_status:
    p_map = p9.ggplot() + p9.geom_blank()
else:
    region_shp = gpd.read_file(region_shp_path)
    p_map = (p9.ggplot() + st.session_state.map_theme +
             p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black"))
    
    if random_status:
        random_locations_shp = gpd.read_file(random_locations_path)
        p_map = (p_map + 
                 p9.geom_map(data = random_locations_shp, mapping = p9.aes(color = 'loc_id')))


# %% Layout

st.title("Sampling region")

main_columns = st.columns([1,1])

# with col1:
#     st.pyplot(p9.ggplot.draw(p_map))
with main_columns[0]:
    st.pyplot(p9.ggplot.draw(p_map))
    

# loc_pt = allpts[allpts.loc_id == loc_id]


loc = gpd.read_file(random_locations_path).set_crs(4326)


st_session_state = {}
if 'loc_id' not in st.session_state:
    st.session_state['loc_id'] = 1
    
loc_id = st.session_state['loc_id']



def next_button():
    class_df_filter = st.session_state.class_df_filter
    current_loc_id = st.session_state.loc_id
    new_locid = class_df_filter.loc_id[class_df_filter['loc_id'] > current_loc_id].min()
    
    # loc_id is max for filters, then cycle back to beginning
    if np.isnan(new_locid):
        new_locid = class_df_filter.loc_id.min()
    st.session_state.loc_id = int(new_locid)
    
def prev_button():
    class_df_filter = st.session_state.class_df_filter
    current_loc_id = st.session_state.loc_id
    new_locid = class_df_filter.loc_id[class_df_filter['loc_id'] < current_loc_id].max()
    
    # loc_id is min for filters, then cycle back to end
    if np.isnan(new_locid):
        new_locid = class_df_filter.loc_id.max()
    st.session_state.loc_id = int(new_locid)

def go_to_id(id_to_go, year):
    st.session_state.loc_id = int(id_to_go)
    st.session_state.classification_year = int(year)
    st.session_state.subclass_year = 'Subclass' + str(year)
    


s1colA, s1colB, s1colC = st.sidebar.columns([3,1,1])

# side_layout = st.sidebar.beta_columns([1,1])
with s1colA: #scol2 # side_layout[-1]:
    st.markdown('### Location ID: ' + str(loc_id))
    # loc_id = int(st.number_input('Location ID', 1, allpts.query('allcomplete').loc_id.max(), 1))
    
    
# %% 
# GO TO EXPANDER

go_to_expander = st.sidebar.expander('Go to')


def go_to_id(id_to_go):
    st.session_state.loc_id = int(id_to_go)

with go_to_expander:
    # st.text('go to year not working')
    s2colA, s2colB = go_to_expander.columns([2,1])

with s1colC:
    st.button('Next', on_click = next_button, args = ())
with s1colB:
    st.button('Prev', on_click = prev_button, args = ())
    
with s2colA:
    id_to_go = st.text_input("ID", value = str(loc_id))
with s2colB:
    st.text("")
    st.text("")
    st.button('Go', on_click = go_to_id, args = (id_to_go, ))
    
# loc_id_num = loc_id
# loc_id = 1
loc_pt = loc[loc.loc_id == loc_id]

st.write(loc_pt)

# %%

adj_y_m = 0
adj_x_m = 30

loc_pt_latlon = [loc_pt.geometry.y, loc_pt.geometry.x]
loc_pt_latlon_shifted = spf.shift_points_m(loc_pt, adj_x_m, adj_y_m)


# loc_pt_latlon = [13, 77]
loc_pt_latlon_adj = [loc_pt_latlon_shifted.geometry.y, loc_pt_latlon_shifted.geometry.x]

# m_folium = folium.Map()
m_folium = folium.Map(location = loc_pt_latlon, zoom_start = 18)
tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
        ).add_to(m_folium)


m_folium \
    .add_child(folium.CircleMarker(location = loc_pt_latlon, radius = 5)) \
    .add_child(folium.CircleMarker(location = loc_pt_latlon_adj, radius = 5, color = 'red'))
    
with main_columns[1]:
    st_folium(m_folium, height = 400, width = 600)

# %%


path_to_shp_import = st.text_input('Path to shapefile',
              value = '/Users/gopal/Projects/ArkavathyTanksProject/arkavathytanks/spatial/CauveryBasin/Cauvery_boundary5.shp')
st.button('Import shapefile', on_click = spf.ImportShapefile, args = (sample_locations_dir_path, path_to_shp_import,))
if region_status: st.text('Already done (' + re.sub(st.session_state.app_path,'',region_shp_path) + ')')
st.button('Generate random locations', on_click = spf.GenerateRandomPts, args = (st.session_state.app_path,st.session_state.proj_name,))
if random_status: st.text('Already done (' + re.sub(st.session_state.app_path,'',random_locations_path) + ')')
st.button('Begin setting sample locations', on_click = spf.InitiateSampleLocations, args = (st.session_state.app_path,st.session_state.proj_name,))
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
    