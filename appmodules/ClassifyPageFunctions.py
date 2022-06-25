#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 08:42:39 2022

@author: gopal
"""

# pageClassify
import streamlit as st


def apply_filter(lat_range, lon_range, class_type, subclass_type, downloaded):
    
    st.session_state['filterargs'] = {
        'lon' : lon_range,
        'lat' : lat_range,
        'Class' : class_type,
        'Subclass' : subclass_type,
        'Downloaded' : 'Yes'
        }
    
    # print(lat_range)
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
    st.session_state.class_df_filter = filterpts
    
    
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