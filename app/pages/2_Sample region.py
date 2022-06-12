#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:03:28 2022

@author: gopal
"""


import streamlit as st
import pandas as pd
import geopandas as gpd
import geemap
import os


path_to_shp_import = st.text_input('Path to shapefile',
              value = '/Users/gopal/Projects/ArkavathyTanksProject/arkavathytanks/spatial/CauveryBasin/Cauvery_boundary5.shp')

shp_path = os.path.join(st.session_state.proj_path,"region")
region_path = os.path.join(shp_path,"region.shp")
# %%

# %%

# import the shapefile to the project directory
def ImportShapefile(x):
    
    st.write('hello world')
    if not os.path.isdir(shp_path): os.mkdir(shp_path)
    if os.path.isfile(region_path):
        st.write('region.shp already exists')
    else:
        region_gpd = gpd.read_file(path_to_shp_import)
        region_gpd.to_file(region_path)
    

# st.button()


st.button('Import shapefile', on_click = ImportShapefile, args = (path_to_shp_import,))

# gpd.read_file()

# foo = pd.DataFrame({'a':[1, 2, 3]})

# st.write(st.session_state.proj_path)

