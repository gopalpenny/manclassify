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
import appmodules.ClassifyPageFunctions as cpf
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from itertools import compress
import math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import OrderedDict


import importlib
importlib.reload(mf)

  
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
#
loc = gpd.read_file(sample_locations_path).set_crs(4326)
ts_status_path = mf.TimeseriesStatusInit(proj_path)

# def plotRegionPoints(region_status, sample_status, ts_status_path, allpts):
ts_status = pd.read_csv(ts_status_path)[['loc_id','allcomplete']]


# %%
if 'classification_year' not in st.session_state:
    st.session_state['classification_year'] = st.session_state['proj_vars']['classification_year_default']
    
# %%
if 'class_df' not in st.session_state:
    st.session_state['class_df'] = mf.InitializeClassDF(class_path, loc)
else:
    st.session_state['class_df'] = mf.InitializeClassDF(class_path, loc)
    
# %%
    
# if 'proj_vars' not in st.session_state:
#     st.session_state['proj_vars'] = mf.readProjectVars(st.session_state['proj_path'])
    
    

    
# %%
# allpts needed to map (ie w geometry) which points have been downloaded
allpts = pd.merge(loc, ts_status, 'outer', on = 'loc_id').merge(st.session_state['class_df'], on = 'loc_id')
allpts['Downloaded'] = pd.Categorical(allpts.allcomplete, categories = [False, True])
allpts['Downloaded'] = allpts.Downloaded.cat.rename_categories(['No','Yes'])
allpts['lat'] = allpts.geometry.y
allpts['lon'] = allpts.geometry.x

st.session_state['allpts'] = allpts


# %%
lon_pts = allpts.geometry.x
lat_pts = allpts.geometry.y

lon_min = float(math.floor(lon_pts.min()))
lon_max = float(math.ceil(lon_pts.max()))
lat_min = float(math.floor(lat_pts.min()))
lat_max = float(math.ceil(lat_pts.max()))


if 'filterargs' not in st.session_state:
    st.session_state['filterargs'] = {
        'lon' : [lon_min, lon_max],
        'lat' : [lat_min, lat_max],
        'Class' : 'Any',
        'Subclass' : 'Any',
        'Downloaded' : 'Yes'
        }

if 'class_df_filter' not in st.session_state:
    filterargs = st.session_state['filterargs']
    # st.session_state['class_df_filter'] = 1#
    
    # class_df_filter set within apply_filter function
    cpf.apply_filter(lat_range = filterargs['lat'], 
                    lon_range = filterargs['lon'], 
                    class_type = filterargs['Class'], 
                    subclass_type = filterargs['Subclass'], 
                    downloaded = filterargs['Downloaded'])
# %%



# %%


st.title("Pixel classification")

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
    


s1colA, s1colB, s1colC = st.sidebar.columns([3,1,1])
# side_layout = st.sidebar.beta_columns([1,1])
with s1colA: #scol2 # side_layout[-1]:
    st.markdown('### Location ID: ' + str(loc_id))
    # loc_id = int(st.number_input('Location ID', 1, allpts.query('allcomplete').loc_id.max(), 1))
    

# %% 
# GO TO EXPANDER

go_to_expander = st.sidebar.expander('Go to')

with go_to_expander:
    st.text('go to year not working')
    s2colA, s2colB, s2colC = go_to_expander.columns([2,2,1])

with s1colC:
    st.button('Next', on_click = next_button, args = ())
with s1colB:
    st.button('Prev', on_click = prev_button, args = ())
    
with s2colA:
    id_to_go = st.text_input("ID", value = str(loc_id))
with s2colB:
    proj_years = st.session_state['proj_vars']['proj_years']
    classification_year = st.session_state['classification_year']
    idx_class_year = [i for i in range(len(proj_years)) if proj_years[i] == classification_year][0]
    # year_to_go = st.text_input("Year", value = str(st.session_state['classification_year']))
    year_to_go = st.selectbox("Year", options = proj_years, index =  idx_class_year)
with s2colC:
    st.text("")
    st.text("")
    st.button('Go', on_click = go_to_id, args = (id_to_go, year_to_go, ))
    
# loc_id_num = loc_id
# loc_id = 1
loc_pt = allpts[allpts.loc_id == loc_id]

# st.write(loc_pt.crs)
# loc_pt_utm = 

adj_y_m = 0
adj_x_m = 30

loc_pt_latlon_shifted = mf.shift_points_m(loc_pt, adj_x_m, adj_y_m)

loc_pt_latlon = [loc_pt.geometry.y, loc_pt.geometry.x]
loc_pt_latlon_adj = [loc_pt_latlon_shifted.geometry.y, loc_pt_latlon_shifted.geometry.x]

# %%

region_shp = gpd.read_file(region_shp_path)
p_map = (p9.ggplot() + 
          p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black") +
          p9.geom_map(data = allpts, mapping = p9.aes(), fill = 'lightgray', shape = 'o', color = None, size = 1, alpha = 1) +
          p9.geom_map(data = st.session_state['class_df_filter'], mapping = p9.aes(fill = 'SubClass'), shape = 'o', color = None, size = 2) +
          p9.geom_map(data = loc_pt, mapping = p9.aes(), fill = 'black', shape = 'o', color = 'black', size = 4) +
          mf.MapTheme() + p9.theme(legend_position = (0.8,0.7)))

# %%
col1, col2 = st.columns(2)

# %%

plot_theme = p9.theme(panel_background = p9.element_rect())

start_date_string_full = (str(st.session_state['classification_year']) + '-' + 
                          st.session_state['proj_vars']['classification_start_month'] + '-' +
                          str(st.session_state['proj_vars']['classification_start_day']))

start_datetime = datetime.strptime(start_date_string_full, '%Y-%B-%d')
end_datetime = start_datetime + relativedelta(years = 1)
datetime_range = [start_datetime, end_datetime]
date_range = [datetime.strftime(x, '%Y-%m-%d') for x in datetime_range]
start_date = date_range[0]
# date_range = ['2019-06-01', '2020-06-01']
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
    # date_start = st.date_input('Date', value = '2019-06-01')
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
              args = (loc_id, ClassBox, SubClass, class_path, new_class, new_subclass, st.session_state['classification_year'], ))



# %%

def next_button():
    st.session_state.loc_id += 1
def prev_button():
    st.session_state.loc_id += 1
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
    .add_child(folium.CircleMarker(location = loc_pt_latlon, radius = 5)) #\
    # .add_child(folium.CircleMarker(location = loc_pt_latlon_adj, radius = 5, color = 'red'))
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



def FilterPts(allpts, lat_range):
    filterpts = st.session_state['allpts']
    # latitude
    filterpts = filterpts[allpts['lat'] >= lat_range[0]]
    filterpts = filterpts[allpts['lat'] <= lat_range[1]]
    # longitude
    filterpts = filterpts[allpts['lon'] >= lon_range[0]]
    filterpts = filterpts[allpts['lon'] <= lon_range[1]]
    return filterpts

# filterpts = FilterPts(allpts, lat_range)

sideexp = st.sidebar.expander('Filter points')
with sideexp:
    se1col1, se1col2 = sideexp.columns([1, 1])
    class_types = ['Any'] + list(st.session_state.class_df.Class.unique())
    cur_class_type = st.session_state['filterargs']['Class']
    class_types_idx = [i for i in range(len(class_types)) if class_types[i] == cur_class_type][0]
    
    with se1col1:
        class_type = st.selectbox('Class (' + cur_class_type + ')', options = class_types, 
                                   index = class_types_idx)
    
    subclass_types = ['Any'] + list(st.session_state.class_df.SubClass.unique())
    cur_subclass_type = st.session_state['filterargs']['Subclass']
    subclass_types_idx = [i for i in range(len(subclass_types)) if subclass_types[i] == cur_subclass_type][0]
    with se1col2:
        subclass_type = st.selectbox('Sub-class (' + cur_subclass_type + ')', options = subclass_types, 
                                   index = subclass_types_idx)
                              
    cur_lat = st.session_state['filterargs']['lat']
    cur_lon = st.session_state['filterargs']['lon']
    lat_header = 'Latitude [' + str(cur_lat[0]) + \
      ', ' + str(cur_lat[1]) + ']'
    lon_header = 'Longitude [' + str(cur_lon[0]) + \
      ', ' + str(cur_lon[1]) + ']'
    # lat_header = 'Latitude ('
    lat_range = st.slider(lat_header, min_value = lat_min, max_value = lat_max, 
              value = (st.session_state['filterargs']['lat'][0], st.session_state['filterargs']['lat'][1]))
    lon_range = st.slider(lon_header, min_value = lon_min, max_value = lon_max, 
              value = (st.session_state['filterargs']['lon'][0], st.session_state['filterargs']['lon'][1]))
    
    download_options = ['All', 'Yes', 'No']
    cur_download_type = st.session_state['filterargs']['Downloaded']
    download_idx = [i for i in range(len(download_options)) if download_options[i] == cur_download_type][0]
    downloaded = st.selectbox('Downloaded (' + cur_download_type + ')', options = download_options, index = download_idx)
    se2col1, se2col2 = sideexp.columns([1, 1])
    
    
with se2col1:
    st.button('Apply filter', on_click=cpf.apply_filter, args = (lat_range, lon_range, class_type, subclass_type, downloaded, ))

with se2col2:
    st.button('Clear filter', on_click=cpf.clear_filter, args = (lat_min, lat_max, lon_min, lon_max, ))
    
    
# %%

# VIEW SNAPSHOTS
start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
end_date = datetime.strptime(date_range[1], '%Y-%m-%d')

# p_s2 = mf.GenS2plot(loc_id, timeseries_dir_path, date_range, plot_theme)
st_snapshots = st.expander('View snapsnots')
stexp1col1, stexp1col2, stexp1col3, stexp1col4 = st_snapshots.columns([1,1,1,1])
tsS2 = mf.GenS2data(loc_id, timeseries_dir_path, date_range).query('cloudmask == 0')
tsS2 = tsS2[start_date <= tsS2['datetime']]
tsS2 = tsS2[tsS2['datetime'] <= end_date]

# dates_datetime = tsS2[datetime.strftime(tsS2['datetime'],'%Y'),'datetime']
dates_datetime = list(OrderedDict.fromkeys(tsS2['datetime']))
dates_str = [datetime.strftime(x, '%Y-%m-%d') for x in dates_datetime]

buffer_px = 10

with stexp1col1:
    # st.write(loc_pt_latlon[1].iloc[0])
    # im_date1 = st.selectbox('Select date 1', options = [dates_str[i] for i in range(len(dates)) if re.sub('.*\\-([0-9]+)\\-.*','\\1',x) in ['01','02','03']])
    im_date1 = st.selectbox('Select date 1', options = [x for x in dates_str if re.sub('.*\\-([0-9]+)\\-.*','\\1',x) in ['01','02','03']])
    im_array1 = cpf.get_image_near_point1('COPERNICUS/S2_SR', im_date1, ['B8','B4','B3'], loc_pt_latlon, buffer_px)
    plt1 = cpf.plot_array_image(im_array1)
    st.pyplot(plt1)
    
with stexp1col2:
    im_date2 = st.selectbox('Select date 2', options = dates_str)
    im_array2 = cpf.get_image_near_point2('COPERNICUS/S2_SR', im_date2, ['B8','B4','B3'], loc_pt_latlon, buffer_px)
    plt2 = cpf.plot_array_image(im_array2)
    st.pyplot(plt2)
    
    

with stexp1col3:
    im_date3 = st.selectbox('Select date 3', options = dates_str)
    im_array3 = cpf.get_image_near_point3('COPERNICUS/S2_SR', im_date3, ['B8','B4','B3'], loc_pt_latlon, buffer_px)
    plt3 = cpf.plot_array_image(im_array3)
    st.pyplot(plt3)
    
with stexp1col4:
    im_date4 = st.selectbox('Select date 4', options = dates_str)
    im_array4 = cpf.get_image_near_point4('COPERNICUS/S2_SR', im_date4, ['B8','B4','B3'], loc_pt_latlon, buffer_px)
    plt4 = cpf.plot_array_image(im_array4)
    st.pyplot(plt4)
    
    
with st.expander('Selected points'):
    # st.write(st.session_state.class_df_filter)
    st.write(pd.DataFrame(st.session_state.class_df_filter).drop('geometry', axis = 1))
    # st.write(st.session_state.class_df_filter)
    # st.dataframe(pd.DataFrame(st.session_state.class_df_filter))
    # st.write(pd.DataFrame(allpts).drop('geometry', axis = 1))