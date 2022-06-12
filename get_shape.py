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

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)

from geemod import rs

ee.Initialize()


shp_path = 

# %%

ic = ee.ImageCollection('COPERNICUS/S2_SR')
im = ic.mosaic()

watershed_pt = ee.Geometry.Point([76.9, 13])

hyd_watershed = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_4') \
  .filterBounds(watershed_pt) \
  .union()
  
# %%
samp_fc = im.sample(
    region = hyd_watershed,
    scale = 10,
    numPixels = 10,
    seed = 10,
    geometries = True).map(rs.set_feature_id_func('loc_id')).select('loc_id')


ee.batch.Export.table.toDrive(
    collection = ,
    description = ,
    fileNamePrefix = ,
    fileFormat = ,
    folder = )