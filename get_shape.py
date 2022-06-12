#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:48:11 2022

@author: gopal
"""

# %% Initialize
# ee_generate_points

import ee
import os
import sys
import geemap

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)

from geemod import rs

ee.Initialize()

# %%



# path_to_shp_import = st.text_input('Path to shapefile',
#               value = '/Users/gopal/Projects/ArkavathyTanksProject/arkavathytanks/spatial/CauveryBasin/Cauvery_boundary5.shp')

# shp_path = os.path.join(st.session_state.proj_path,"region")
app_path = '/Users/gopal/Google Drive/_Research projects/ML/manclassify/app_data'
proj_name = 'region1'
proj_path = os.path.join(proj_path,proj_name)
region_path = os.path.join(proj_path,"region1")
region_shp_path = os.path.join(proj_path,"region.shp")

region_shp = gpd.read_file(region_shp_path)


region_fc_full = geemap.geopandas_to_ee(region_shp)
region_fc = region_fc_full.union()

# %%

ic = ee.ImageCollection('COPERNICUS/S2_SR')
im = ic.mosaic()

# watershed_pt = ee.Geometry.Point([76.9, 13])

# hyd_watershed = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_4') \
#   .filterBounds(watershed_pt) \
#   .union()

# path = 

folder = proj_name + '_manclassify_pts'
  
# %%
samp_fc = im.sample(
    region = region_fc,
    scale = 10,
    numPixels = 10,
    seed = 10,
    geometries = True).map(rs.set_feature_id_func('loc_id')).select('loc_id')


task = ee.batch.Export.table.toDrive(
    collection = samp_fc,
    description = 'Generating samples',
    fileNamePrefix = 'region_samples',
    fileFormat = 'SHP',
    folder = folder)

task.start()