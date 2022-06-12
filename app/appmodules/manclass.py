#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:10:02 2022

@author: gopal
"""
import streamlit as st
import os
import geopandas as gpd
# appmodule

def testfunc():
    st.write("success")
    
    
# import the shapefile to the project directory
def ImportShapefile(region_path, path_to_shp_import):
    
    region_shp_path = os.path.join(region_path,"region.shp")
    # st.write('hello world')
    if not os.path.isdir(region_path): os.mkdir(region_path)
    if os.path.isfile(region_shp_path):
        st.write('region.shp already exists')
    else:
        region_gpd = gpd.read_file(path_to_shp_import)
        region_gpd.to_file(region_path)
        

def GenerateSamples(app_path, proj_name):
    
    # Generate subdirectories
    proj_path = os.path.join(app_path,proj_name)
    region_path = os.path.join(proj_path,"region")
    region_shp_path = os.path.join(region_path,"region.shp")
    
    
    samples_dir_name = proj_name + '_sample_locations'
    samples_dir_path = os.path.join(proj_path,samples_dir_name)
    
    samples_name = 'sample_locations'
    samples_path = os.path.join(samples_dir_path,samples_name + '.shp')
    
    # %% GENERATE THE SAMPLES IF sample_locations.shp DOES NOT EXIST
    if os.path.exists(samples_path):
        st.write(samples_name + '.shp already exists')
        
    else:
        if not os.path.exists(samples_dir_path): 
            os.mkdir(samples_dir_path)
            
        # Initialize earth engine
        ee.Initialize()
        
        # Import region.shp
        region_shp = gpd.read_file(region_shp_path)
        
        # Convert region.shp to ee.FeatureCollection
        region_fc_full = geemap.geopandas_to_ee(region_shp)
        region_fc = region_fc_full.union()
        
        # Get Image Collection for sample locations
        ic = ee.ImageCollection('COPERNICUS/S2_SR')
        im = ic.mosaic()
            
        # Generate the sample 
        samp_fc = im.sample(
            region = region_fc,
            scale = 10,
            numPixels = 1000,
            seed = 10,
            geometries = True).map(rs.set_feature_id_func('loc_id')).select('loc_id')
        
        # Export the sample
        task = ee.batch.Export.table.toDrive(
            collection = samp_fc,
            description = 'Generating sample locations',
            fileNamePrefix = samples_name,
            fileFormat = 'SHP',
            folder = samples_dir_name)
        
        task.start()
        
        st.write("Sent task to Earth Engine")