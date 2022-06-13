#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:10:02 2022

@author: gopal
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import ee
import sys
import re
# import ?

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemod import rs
from geemod import eesentinel as ees
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
        
# %%


def DownloadPoints(loc, date_range, timeseries_dir_path, ts_status):
    """
    This function downloads all the points in loc using DownloadSamplePt()

    Parameters
    ----------
    loc : gpd.DataFrame
        GeoPandas dataframe to be downloaded
    date_range : LIST (STR)
        Start date and end date as ['YYYY-MM-DD', 'YYYY-MM-DD'].
    timeseries_dir_path : STR
        Path to the google drive timeseries directory where output will be stored.

    Returns
    -------
    None.

    """
    
    print('Downloading ' + str(loc.shape[0]) + ' points')

    for i in range(loc.shape[0]):
        # print(i)
        # i = 1
        pt_gpd = loc.iloc[i]
        sample_pt_coords = [pt_gpd.geometry.x, pt_gpd.geometry.y]
        
        loc_id = loc.loc_id.iloc[i]
        DownloadSamplePt(sample_pt_coords, loc_id, timeseries_dir_path, date_range)



def DownloadSamplePt(sample_pt_coords, loc_id, timeseries_dir_path, date_range):
    """
    This function is used to sample imagery using Google Earth Engine
    The point coordinate is used to generate timeseries within the date_range
    and export the results to Google Drive. It runs one point at a time. 
    Intended to be used within a for loop or mapped over a list of points.

    Parameters
    ----------
    sample_pt_coords : list (float)
        List of length 2 as [x, y] coordinates.
    loc_id : INT
        loc_id for the point
    timeseries_dir_path : str
        path to the directory where results will be saved /proj_name/proj_name_pt_timeseries.
    date_range : list (str)
        List of length 2 as [start_date, end_date] for downloading data.

    Returns
    -------
    None.

    """
    sample_pt_name = 'pt_ts_loc' + str(loc_id)
    
    sample_pt = ee.Geometry.Point(sample_pt_coords)
    
    timeseries_dir_name = re.sub('.*/(.*)', '\\1', timeseries_dir_path)
    
    # Export S1
    s1_colname = 'pt_ts_loc_s1'
    s1_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', s1_colname) #sample_pt_name + '_' + s1_colname
    s1_pt_filepath = os.path.join(timeseries_dir_path, s1_pt_filename + '.csv')
    
    if os.path.exists(s1_pt_filepath):
        print(s1_pt_filename + '.csv already exists')
        st.write(s1_pt_filename + '.csv already exists')
    else:
        s1_output_bands = ['HH','VV','HV','VH','angle']
        s1_ic = ee.ImageCollection("COPERNICUS/S1_GRD") \
          .filterBounds(sample_pt) \
          .filterDate(date_range[0],date_range[1])
          
        # Get S1 pixel timeseries
        s1_ts = rs.get_pixel_ts_allbands(
            pts_fc = ee.FeatureCollection(sample_pt),
            image_collection = s1_ic,
            ic_property_id = 'system:index',
            scale = 10) # for Landsat resolution
        # time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
            
        task_s1 = ee.batch.Export.table.toDrive(
            collection = s1_ts,
            selectors = s1_output_bands + ['image_id'],
            folder = timeseries_dir_name,
            description = s1_pt_filename,
            fileNamePrefix = s1_pt_filename)
        
        task_s1.start()
        
        TimeseriesUpdateStatus(loc_id, s1_colname, 'Running', timeseries_dir_path)
        
        print('Generating ' + s1_pt_filename + '.csv')
        st.write('Generating ' + s1_pt_filename + '.csv')
      
    # Export S2
    
    s2_colname = 'pt_ts_loc_s2'
    s2_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', s2_colname) #sample_pt_name + '_s2'
    s2_pt_filepath = os.path.join(timeseries_dir_path, s2_pt_filename + '.csv')
    
    if os.path.exists(s2_pt_filepath):
        print(s2_pt_filename + '.csv already exists')
        st.write(s2_pt_filename + '.csv already exists')
    else:
    
        s2_output_bands = ['B8','B4','B3','B2','clouds','cloudmask','shadows','probability']
        
        # params variable is used to pass  information to the cloud masking functions.
        # see help(add_cld_shadow_mask_func)
        s2params = {
            'START_DATE' : date_range[0],
            'END_DATE' : date_range[1],
            'CLOUD_FILTER' : 50,
            'CLD_PRB_THRESH' : 53, # 53 for Cauvery # 55 for Indus
            'NIR_DRK_THRESH' : 0.2,
            'CLD_PRJ_DIST' : 1,
            'BUFFER' : 50
        }
        
        s2_clouds_ic = ees.get_s2_sr_cld_col(sample_pt, s2params) \
          .map(ees.add_cld_shadow_mask_func(s2params))
        
        # For some reason the reproject() works so that subsequent sampling returns the whole rectangular array
        # see https://stackoverflow.com/questions/64012752/gee-samplerectangle-returning-1x1-array
        # s2_clouds_im = s2_clouds_ic.mosaic().reproject(crs = ee.Projection('EPSG:4326'), scale=10) #.clip(hyd_watershed)
        
        # Get pixel timeseries
        s2_ts = rs.get_pixel_ts_allbands(
            pts_fc = ee.FeatureCollection(sample_pt),
            image_collection = s2_clouds_ic,
            ic_property_id = 'system:index',
            scale = 10) # for Landsat resolution
        # time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
            
        task_s2 = ee.batch.Export.table.toDrive(
            collection = s2_ts,
            selectors = s2_output_bands + ['image_id'],
            folder = timeseries_dir_name,
            description = s2_pt_filename,
            fileNamePrefix = s2_pt_filename)
        
        task_s2.start()
        
        TimeseriesUpdateStatus(loc_id, s2_colname, 'Running', timeseries_dir_path)
        
        print('Generating ' + s2_pt_filename + '.csv')
        st.write('Generating ' + s2_pt_filename + '.csv')
        
        
        
# %% TIME SERIES STATUS



def TimeseriesStatusInit(proj_path):
    proj_name = re.sub('.*/(.*)', '\\1', proj_path)
    timeseries_dir_path = os.path.join(proj_path, proj_name + '_pt_timeseries')

    # Create timeseries directory if it doesn't exist
    if not os.path.exists(timeseries_dir_path): os.mkdir(timeseries_dir_path)
    
    # Generate status path
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    
    # If it doesn't exist, create a blank file with ts_status
    if not os.path.exists(ts_status_path):
        sample_locations_path = os.path.join(proj_path, proj_name + "_sample_locations/sample_locations.shp")
        loc = gpd.read_file(sample_locations_path)
        # loc[['loc_id']]
        ts_status = pd.DataFrame({'loc_id' : loc.loc_id})
        ts_status['allcomplete'] = False
        
        ts_status.to_csv(ts_status_path, index= False)
    
    return ts_status_path
        

def rowStatus(rowList):
    """Helper function for TimeseriesUpdateStatus
    Checks to see if a csv file is available for all output files
    """
    val = all(['.csv' in str(x) for x in rowList])
    return val

def TimeseriesUpdateStatus(loc_id, colname, new_status, timeseries_dir_path):
    """
    Update the status of a specific loc_id and colname

    Parameters
    ----------
    loc_id : INT
        ID of location.
    colname : STR
        name of column to update.
    new_status : STR
        description of updated status.
    proj_path : STR
        path to the project.

    Returns
    -------
    None.

    """
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    ts_status = pd.read_csv(ts_status_path)
    idx = ts_status.index[ts_status.loc_id == loc_id]
    
    if not colname in ts_status.columns:
        ts_status[colname] = np.nan
        
    ts_status.loc[idx, colname] = new_status
    
    ts_status['allcomplete'] = ts_status.drop(['loc_id','allcomplete'], axis = 1).apply(rowStatus, axis = 1)
    
    ts_status.to_csv(ts_status_path)