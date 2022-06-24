#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:40:03 2022

@author: gopal
"""

import streamlit as st
st.set_page_config(page_title="Classify", layout="wide", page_icon="🌍")
import pandas as pd
import numpy as np
import os
import re
import plotnine as p9
import leafmap
import appmodules.manclass as mf
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from itertools import compress
import math

  
gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# sys.path.append(gdrive_ml_path)
out_folder = 'gee_sentinel_ts'
data_path = os.path.join(gdrive_ml_path, 'manclassify/script_output', out_folder)


# %%

# %%
app_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data'
proj_name = 'region1'
proj_path = os.path.join(app_path, proj_name)

sample_locations_dir_path = os.path.join(proj_path,proj_name + "_sample_locations")
sample_locations_path = os.path.join(sample_locations_dir_path, "sample_locations.shp")
region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")


region_status = os.path.exists(region_shp_path)
sample_status = os.path.exists(sample_locations_path)

timeseries_dir_name = proj_name + "_download_timeseries"
timeseries_dir_path = os.path.join(proj_path, timeseries_dir_name)

classification_dir_name = proj_name + "_classification"
classification_dir_path = os.path.join(proj_path, classification_dir_name)
if not os.path.exists(classification_dir_path): os.mkdir(classification_dir_path)

class_path = os.path.join(classification_dir_path, 'location_classification.csv')
    
    
# %%
import importlib
importlib.reload(mf)
#
loc = gpd.read_file(sample_locations_path)
ts_status_path = mf.TimeseriesStatusInit(proj_path)

# def plotRegionPoints(region_status, sample_status, ts_status_path, allpts):
ts_status = pd.read_csv(ts_status_path)[['loc_id','allcomplete']]

# %%
if not 'class_df' in st.session_state:
    st.session_state['class_df'] = mf.InitalizeClassDF(class_path, loc)
else:
    st.session_state['class_df'] = mf.InitalizeClassDF(class_path, loc)

# %%
# allpts needed to map (ie w geometry) which points have been downloaded
allpts = pd.merge(loc, ts_status, 'outer', on = 'loc_id').merge(st.session_state['class_df'], on = 'loc_id')
allpts['Downloaded'] = pd.Categorical(allpts.allcomplete, categories = [False, True])
allpts['Downloaded'] = allpts.Downloaded.cat.rename_categories(['No','Yes'])
allpts['lat'] = allpts.geometry.y
allpts['lon'] = allpts.geometry.x
# %%

# filterargs = {
#     'lon' : [-180, 180],
#     'lat' : [-90, 90],
#     'Class' : 'Any',
#     'Subclass' : 'Any',
#     'Downloaded' : 'Yes'
#     }

def FilterPts(allpts, lat_range):
    filterpts = allpts
    # latitude
    filterpts = filterpts[allpts['lat'] >= lat_range[0]]
    filterpts = filterpts[allpts['lat'] <= lat_range[1]]
    # longitude
    filterpts = filterpts[allpts['lon'] >= lon_range[0]]
    filterpts = filterpts[allpts['lon'] <= lon_range[1]]
    return filterpts

# filterpts = FilterPts(allpts, lat_range)


# %%


st.title("Pixel classification")

st_session_state = {}
st_session_state['loc_id'] = 1

# side_layout = st.sidebar.beta_columns([1,1])
with st.sidebar: #scol2 # side_layout[-1]:
    loc_id = int(st.number_input('Location ID', 1, allpts.query('allcomplete').loc_id.max(), 1))
    
# loc_id_num = loc_id
# loc_id = 1
loc_pt = allpts[allpts.loc_id == loc_id]
loc_pt_latlon = [loc_pt.geometry.y, loc_pt.geometry.x]

adj_y_m = 10 / 1e3 / 111
adj_x_m = 10 / 1e3 / 111
loc_pt_latlon_adj = [loc_pt.geometry.y + adj_y, loc_pt.geometry.x + adj_x]

# %%

region_shp = gpd.read_file(region_shp_path)
p_map = (p9.ggplot() + 
          p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black") +
          p9.geom_map(data = allpts, mapping = p9.aes(fill = 'Downloaded'), shape = 'o', color = None, size = 2) +
          p9.geom_map(data = loc_pt, mapping = p9.aes(), fill = 'black', shape = 'o', color = 'black', size = 4) +
          mf.MapTheme() + p9.theme(legend_position = (0.8,0.7)))

# %%
col1, col2 = st.columns(2)

# %%

plot_theme = p9.theme(panel_background = p9.element_rect())
date_range = ['2019-06-01', '2021-06-01']
# p_s1 = mf.GenS1plot(loc_id, timeseries_dir_path, date_range, plot_theme)
# p_s2 = mf.GenS2plot(loc_id, timeseries_dir_path, date_range, plot_theme)
p_sentinel = mf.plotTimeseries(loc_id, timeseries_dir_path, date_range)

# GetLocData


# %%

# list(allpts.loc[allpts.loc_id == loc_id, 'Downloaded'])[0]


# %%
# st.write(st.session_state.class_df)
Class_prev = list(st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'Class'])[0]
Classes =  list(st.session_state.class_df.Class.unique()) + ['Input new']
Classes = [x for x in Classes if x != '-']
Classes = ['-'] + list(compress(Classes, [str(x) != 'nan' for x in Classes]))
Classesidx = [i for i in range(len(Classes)) if Classes[i] == Class_prev] + [0]

new_class = "-"


scol1, scol2 = st.sidebar.columns([1,1])




with scol1:
    ClassBox = st.selectbox("Class: " + str(Class_prev), 
                 options = Classes, 
                 index = Classesidx[0])
    if ClassBox == 'Input new':
        new_class = st.text_input('New Class')
        

SubClass_prev = list(st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'SubClass'])[0]
# SubClasses = list(st.session_state.class_df.SubClass.unique()) + ['Input new']
Class_subset = st.session_state.class_df #[Class_subset.Class == ClassBox]
SubClasses = list(Class_subset.SubClass.unique()) + ['Input new']
SubClasses = [x for x in SubClasses if x != '-']
SubClasses = ['-'] + list(compress(SubClasses, [str(x) != 'nan' for x in SubClasses]))
SubClassesidx = [i for i in range(len(SubClasses)) if SubClasses[i] == SubClass_prev] + [0]
new_subclass = "-"
        
with scol2:
    SubClass = st.selectbox("Sub-class: " + str(SubClass_prev), 
                 options = SubClasses, 
                 index = SubClassesidx[0])
    if SubClass == 'Input new':
        new_subclass = st.text_input('New Sub-class')
        
with st.sidebar:
    st.button('Update classification', on_click = mf.UpdateClassDF,
              args = (loc_id, ClassBox, SubClass, class_path, new_class, new_subclass,))



# %%
lon_pts = allpts.geometry.x
lat_pts = allpts.geometry.y

lon_min = float(math.floor(lon_pts.min()))
lon_max = float(math.ceil(lon_pts.max()))
lat_min = float(math.floor(lat_pts.min()))
lat_max = float(math.ceil(lat_pts.max()))


# %%

def next_button():
    st.session_state.counter += 1
def prev_button():
    st.session_state.counter -= 1
# with scol1: #side_layout[0]:
#     st.text(' ')
#     st.text(' ')
#     st.button('Prev', on_click = prev_button, args = ())
# with scol3: #side_layout[-1]:
#     st.text(' ')
#     st.text(' ')
#     st.button('Next', on_click = next_button, args = ())
    
with st.sidebar:
    st.pyplot(p9.ggplot.draw(p_map))

# with st.sidebar:
#     lat = st.number_input('Location ID', 0.0, 90.0, 13.0, ) #, step = 0.1)
#     lon = st.number_input('Lon', -180.0, 180.0, 77.0) #, step = 0.1)
    # st.write(data_path_files)
    
# m = leafmap.Map(center=(lat, lon), zoom=18)
# m.add_tile_layer(
#     url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
#     name="Google Satellite",
#     attribution="Google",
# )

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
# point = 
# tile1 = folium.TileLayer(
#         tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
#         attr = 'Google',
#         name = 'Google Satellite',
#         overlay = False,
#         control = True
#        ).add_to(m)



# with col1:
#     st.text("Col 1")
    
with col1:
    st.pyplot(p9.ggplot.draw(p_sentinel))
    # st.pyplot(p9.ggplot.draw(p_s2))
    
with col2:
    # st.write(m.to_streamlit())
    st_folium(m_folium, height = 400, width = 600)

# %%

# with st.sidebar:

# sideexp = st.sidebar.expander('Filter points')
# with sideexp:
#     lat_range = st.slider('Longitude', min_value = lon_min, max_value = lon_max, 
#               value = (lon_min, lon_max))
#     lon_range = st.slider('Latitude', min_value = lat_min, max_value = lat_max, 
#               value = (lat_min, lat_max))
#     s2col1, s2col2 = sideexp.columns([1, 1])
    
# with s2col1:
#     st.button('Apply filter')

# with s2col2:
#     st.button('Clear filter')
    
with st.expander('Selected points'):
    st.write(pd.DataFrame(allpts).drop('geometry', axis = 1))