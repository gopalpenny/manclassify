#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 08:42:39 2022

@author: gopal
"""

# pageClassify
import streamlit as st
from datetime import datetime, timedelta
import ee
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import math

def apply_filter(lat_range, lon_range, class_type, subclass_type, downloaded):
    
    st.session_state['filterargs'] = {
        'lon' : lon_range,
        'lat' : lat_range,
        'Class' : class_type,
        'Subclass' : subclass_type,
        'Downloaded' : 'Yes'
        }
    
    filterpts = st.session_state['allpts']
    
    # class
    if class_type != 'Any':
        filterpts = filterpts[filterpts['Class'] == class_type]
        
    # subclass
    if subclass_type != 'Any':
        filterpts = filterpts[filterpts['SubClass'] == subclass_type]
        
        
    # filter for downloaded points
    if subclass_type != 'All':
        filterpts = filterpts[filterpts['Downloaded'] == downloaded]
    
    # latitude
    filterpts = filterpts[filterpts['lat'] >= lat_range[0]]
    filterpts = filterpts[filterpts['lat'] <= lat_range[1]]
    # print(filterpts)
    # longitude
    filterpts = filterpts[filterpts['lon'] >= lon_range[0]]
    filterpts = filterpts[filterpts['lon'] <= lon_range[1]]
    st.session_state['class_df_filter'] = filterpts
    
    
def clear_filter(lat_min, lat_max, lon_min, lon_max):
    filterpts = st.session_state['allpts']
    st.session_state['filterargs'] = {
        'lon' : [lon_min, lon_max],
        'lat' : [lat_min, lat_max],
        'Class' : 'Any',
        'Subclass' : 'Any',
        'Downloaded' : 'Yes'
        }
    st.session_state.class_df_filter = filterpts
    
    
# %% 
# @st.cache
def get_image_near_point(im_collection_id, im_date,  bands_rgb, latitude, longitude, buffer_m, 
                        return_geopandas = False):
    
    start_datetime = datetime.strptime(im_date,'%Y-%m-%d')
    end_date = datetime.strftime(start_datetime + timedelta(days = 1), '%Y-%m-%d')
    
    try:
        pt = ee.Geometry.Point([longitude, latitude])
    except:
        ee.Initialize()
        pt = ee.Geometry.Point([longitude, latitude])
        
    pt_bbox = pt.buffer(buffer_m, 1).bounds()
    ic = ee.ImageCollection(im_collection_id).filterBounds(pt).filterDate(im_date, end_date)
    # ic = ic.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 0.25))
    im = ic.first().select(bands_rgb)

    if return_geopandas:
        im = im.addBands(im.pixelCoordinates(im.projection()))

    # sample the region and return array from ee to python
    im_dict = im.sampleRectangle(region = pt_bbox, properties = []).getInfo()

    # len(im_dict['properties'])
    im_props = im_dict['properties']
    im_props_keys = list(im_props.keys())

    if return_geopandas:
        df = pd.DataFrame()
        for i in range(len(im_props)):
            colname = im_props_keys[i]
            data = np.array(im_props[colname]).flatten()
            df[colname] = data

        im_projection = im.projection().getInfo()
        # convert to geopandas
        gdf = gpd.GeoDataFrame(df, 
                 geometry = gpd.points_from_xy(df.x, df.y),
                 crs = im_projection['crs'])
        
        return_val = gdf


    else:
        # extract each band separately
        Bred = np.expand_dims(np.array(im_props[bands_rgb[0]]), 2)
        Bgreen = np.expand_dims(np.array(im_props[bands_rgb[1]]), 2)
        Bblue = np.expand_dims(np.array(im_props[bands_rgb[2]]), 2)

        im_array_rgb = np.concatenate((Bred, Bgreen, Bblue), axis = 2)
        return_val = im_array_rgb
    
    return return_val

# @st.cache
def plot_array_image(im_array):
    xcenter = math.ceil(im_array.shape[0] / 2)
    ycenter = math.ceil(im_array.shape[1] / 2)
    
    maxval = 5000
    
    im_array[im_array > maxval] = maxval
    
    arrow_spacing = 1.5
    # Scale the data to [0, 255] to show as an RGB image.
    rgb_img_test = (255*((im_array - 0)/5000)).astype('uint8')
    plt.figure(figsize = (5,5),dpi = 100)
    plt.axis('off')
    plt.imshow(rgb_img_test)
    plt.plot(xcenter, ycenter - arrow_spacing, marker = 'v', color = 'white')
    plt.plot(xcenter, ycenter + arrow_spacing,marker = '^', color = 'white')
    plt.plot(xcenter + arrow_spacing, ycenter, marker = '<', color = 'white')
    plt.plot(xcenter - arrow_spacing, ycenter,marker = '>', color = 'white')
    return plt