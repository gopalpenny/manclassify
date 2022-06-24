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
from itertools import compress
import plotnine as p9
import math
import geemap
# import ?

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research/Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemod import rs
from geemod import eesentinel as ees
# appmodule

def testfunc():
    st.write("success")
    
    
# import the shapefile to the project directory
def ImportShapefile(sample_locations_dir_path, path_to_shp_import):
    
    region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")
    # st.write('hello world')
    if not os.path.isdir(sample_locations_dir_path): os.mkdir(sample_locations_dir_path)
    if os.path.isfile(region_shp_path):
        st.write('region.shp already exists')
    else:
        region_gpd = gpd.read_file(path_to_shp_import)
        region_gpd.to_file(sample_locations_dir_path)
        

def GenerateSamples(app_path, proj_name):
    
    # Generate subdirectories
    proj_path = os.path.join(app_path,proj_name)
    sample_locations_dir_path = os.path.join(proj_path, proj_name + "_sample_locations")
    region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")
    
    
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
    
    # TimeseriesUpdateAllStatus(timeseries_dir_path)
    
    print('Downloading ' + str(loc.shape[0]) + ' points')
    pbar = st.progress(0)
    infobox = st.empty()

    for i in range(loc.shape[0]):
        # print(i)
        # i = 1
        pt_gpd = loc.iloc[i]
        sample_pt_coords = [pt_gpd.geometry.x, pt_gpd.geometry.y]
        
        loc_id = loc.loc_id.iloc[i]
        DownloadSamplePt(sample_pt_coords, loc_id, timeseries_dir_path, date_range, infobox)
        pbar.progress((float(i) + 1) / loc.shape[0])
        
        
    st.success("All sample locations are now being processed by Google Earth Engine")



def DownloadSamplePt(sample_pt_coords, loc_id, timeseries_dir_path, date_range, infobox):
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
        path to the directory where results will be saved /proj_name/proj_name_download_timeseries.
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
    
    s1_pt_status = TimeseriesCheckStatus(loc_id, s1_colname, timeseries_dir_path)
    if os.path.exists(s1_pt_filepath):
        dummyvariable = s1_pt_filepath
        infobox.info(s1_pt_filename + '.csv already exists')
        # st.write(s1_pt_filename + '.csv already exists')
    elif str(s1_pt_status) != 'nan':
        msgs1 = s1_pt_filename + ' status is ' + str(s1_pt_status)
        infobox.info(msgs1)
        # st.write(msgs1)
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
        
        TimeseriesUpdateLocStatus(loc_id, s1_colname, 'Running', timeseries_dir_path)
        
        infobox.info('Generating ' + s1_pt_filename + '.csv')
        # st.write('Generating ' + s1_pt_filename + '.csv')
      
    # Export S2
    
    s2_colname = 'pt_ts_loc_s2'
    s2_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', s2_colname) #sample_pt_name + '_s2'
    s2_pt_filepath = os.path.join(timeseries_dir_path, s2_pt_filename + '.csv')
    
    s2_pt_status = TimeseriesCheckStatus(loc_id, s2_colname, timeseries_dir_path)
    if os.path.exists(s2_pt_filepath):
        dummy = s2_pt_filepath
        # print(s2_pt_filename + '.csv already exists')
        infobox.info(s2_pt_filename + '.csv already exists')
        
    elif s2_pt_status != 'nan':
        msgs2 = s2_pt_filename + ' status is ' + str(s2_pt_status)
        # print(msgs2)
        infobox.info(msgs2)
        # st.write(type(s2_pt_status))
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
        
        TimeseriesUpdateLocStatus(loc_id, s2_colname, 'Running', timeseries_dir_path)
        
        # print('Generating ' + s2_pt_filename + '.csv')
        infobox.info('Generating ' + s2_pt_filename + '.csv')
        
        
        
# %% TIME SERIES STATUS



def TimeseriesStatusInit(proj_path):
    proj_name = re.sub('.*/(.*)', '\\1', proj_path)
    timeseries_dir_path = os.path.join(proj_path, proj_name + '_download_timeseries')

    # Create timeseries directory if it doesn't exist
    if not os.path.exists(timeseries_dir_path): os.mkdir(timeseries_dir_path)
    
    # Generate status path
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    
    # If it doesn't exist, create a blank file with ts_status
    if not os.path.exists(ts_status_path):
        
        ts_all_files = os.listdir(timeseries_dir_path)
        ts_all_filenames = list(set([re.sub('.csv','',re.sub('loc[0-9]+_','loc_',x)) for x in ts_all_files]))
        file_colnames = list(compress(ts_all_filenames, ['loc' in x for x in ts_all_filenames]))
        
        sample_locations_path = os.path.join(proj_path, proj_name + "_sample_locations/sample_locations.shp")
        loc = gpd.read_file(sample_locations_path)
        # loc[['loc_id']]
        ts_status = pd.DataFrame({'loc_id' : loc.loc_id})
        ts_status['allcomplete'] = False
        
        for colname in file_colnames:
            ts_status[colname] = np.nan
        
        ts_status.to_csv(ts_status_path, index= False)
    
    TimeseriesUpdateAllStatus(timeseries_dir_path)
    
    return ts_status_path
        

def rowStatus(rowList):
    """Helper function for TimeseriesUpdateLocStatus
    Checks to see if a csv file is available for all output files
    """
    val = all(['.csv' in str(x) for x in rowList])
    return val

def TimeseriesUpdateLocStatus(loc_id, colname, new_status, timeseries_dir_path):
    """
    Update the status of a specific loc_id and colname

    Parameters
    ----------
    loc_id : INT
        ID of location.
    colname : STR
        name of column to update.
    new_status : STR
        description of updated status. if 'check', update status if file exists
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
        
    # # Check to 
    # if new_status == 'check':
    #     loc_column_csv_filename = re.sub('loc_','loc' + str(loc_id) + '_',colname) + '.csv'
    #     loc_column_csv_path = os.path.join(timeseries_dir_path, loc_column_csv_filename)
    #     if os.path.exists(loc_column_csv_path):
    #         ts_status.loc[idx, colname] = loc_column_csv_filename
    # else:
    ts_status.loc[idx, colname] = new_status
        
    ts_status.to_csv(ts_status_path, index = False)
    
# %%
foo = 'hi'

if foo == 'hi':
    print('bye')
# %%

def TimeseriesUpdateAllStatus(timeseries_dir_path):
    """
    Update the status of all loc_id's by checking for .csv files for every colname

    Parameters
    ----------
    timeseries_dir_path : STR
        path to the timeseries subdirectory.

    Returns
    -------
    None.

    """
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    ts_status = pd.read_csv(ts_status_path)
    
    
    all_loc = ts_status.loc_id
    all_columns = ts_status.columns
    
    
    for loc_id in all_loc:
        for colname in all_columns:
            loc_column_csv_filename = re.sub('loc_','loc' + str(loc_id) + '_',colname) + '.csv'
            loc_column_csv_path = os.path.join(timeseries_dir_path, loc_column_csv_filename)
            if os.path.exists(loc_column_csv_path):
                idx = ts_status.index[ts_status.loc_id == loc_id]
                ts_status.loc[idx, colname] = loc_column_csv_filename
            # TimeseriesUpdateLocStatus(loc_id, colname, 'check', timeseries_dir_path)
    
    ts_status['allcomplete'] = ts_status.drop(['loc_id','allcomplete'], axis = 1).apply(rowStatus, axis = 1).to_list()
    
    ts_status.to_csv(ts_status_path, index = False)
    

    
def TimeseriesCheckStatus(loc_id, colname, timeseries_dir_path):
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
    
    return str(ts_status.loc[idx, colname].to_list()[0])




# def GetLocTimeseries(loc_id, timeseries_dir_path, plot_theme):
def GenS1data(loc_id, timeseries_dir_path, date_range):
    s1_filename = 'pt_ts_loc' + str(loc_id) + '_s1.csv'
    s1 = pd.read_csv(os.path.join(timeseries_dir_path,s1_filename))
    
    s1['backscatter'] = (s1['VV']**2 + s1['VH']**2) ** (1/2)
    
    s1['datestr'] = [re.sub('.*?_1SDV_([0-9T]+)_.*','\\1',x) for x in s1['image_id']]
    
    s1['datetime'] = pd.to_datetime(s1['datestr'])
    
    s1_long = s1.melt(id_vars = 'datetime', value_vars = 'backscatter')
    
    return s1_long
    


    
    # p_s1 = (p9.ggplot(data = s1, mapping = p9.aes('datetime', 'backscatter')) + 
    #   p9.geom_point() + 
    #   p9.geom_smooth(span = 0.25) + 
    #   # p9.xlim()+
    #   # p9.scale_x_datetime(limits = [datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)], 
    #   p9.scale_x_datetime(limits = pd.to_datetime(date_range), 
    #                       date_labels = '%Y-%b', date_breaks = '1 year') +
    #   plot_theme)
    
    # return p_s1

def GenS2data(loc_id, timeseries_dir_path, date_range):
    s2_filename = 'pt_ts_loc' + str(loc_id) + '_s2.csv'
    s2 = pd.read_csv(os.path.join(timeseries_dir_path,s2_filename))
        
    # time_series_pd['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in time_series_pd_load['image_id']]
    s2['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in s2['image_id']]
    
    s2['datetime'] = pd.to_datetime(s2['datestr'])
    s2 = s2.assign(NDVI = lambda df: (df.B8 - df.B4)/(df.B8 + df.B4))
    
    
    s2_long = s2.melt(id_vars = ['datetime','cloudmask'], value_vars = 'NDVI')
    
    return s2_long
    
    # # s2['backscatter'] = (s2['VV']**2 + s2['VH']**2) ** (1/2)

def plotTimeseries(loc_id, timeseries_dir_path, date_range):
    
    s1 = GenS1data(loc_id, timeseries_dir_path, date_range)
    s2 = GenS2data(loc_id, timeseries_dir_path, date_range)
    
    sentinel = pd.concat([s1, s2])
    

    line_vars = ['NDVI']
    smooth_vars = ['backscatter']
    sentinel_cloudfree = sentinel.query('cloudmask != 1')
    
    p_sentinel = (p9.ggplot(data = sentinel_cloudfree, mapping = p9.aes('datetime', 'value')) + 
      p9.geom_point() + 
      p9.geom_line(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(line_vars)]) + 
      p9.geom_smooth(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(smooth_vars)], span = 0.25) + 
      p9.facet_wrap('variable',scales = 'free_y',ncol = 1) +
      # p9.xlim()+
      # p9.scale_x_datetime(limits = [datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)], 
      p9.scale_x_datetime(limits = pd.to_datetime(date_range), 
                          date_labels = '%Y-%b', date_breaks = '1 year') +
      PlotTheme() + p9.theme(axis_title_x = p9.element_blank()))
    
    return p_sentinel


def MapTheme():
    map_theme = p9.theme(panel_background = p9.element_rect(fill = None),      
                     panel_border = p9.element_rect(),
                     panel_grid_major=p9.element_blank(),
                     panel_grid_minor=p9.element_blank(),
                     plot_background=p9.element_rect(fill = None))
    return map_theme




def PlotTheme():
    strlit_color = '#0F1116'
    plot_theme = p9.theme(panel_background = p9.element_rect(fill = None),      
                     panel_border = p9.element_rect(color = None),
                     panel_grid_major=p9.element_blank(),
                     panel_grid_minor=p9.element_blank(),
                     axis_text = p9.element_text(color = 'white'),
                     axis_ticks = p9.element_line(color = 'white'),
                     axis_title = p9.element_text(color = 'white'),
                     # plot_background=p9.element_rect(fill = 'black'),
                     plot_background=p9.element_rect(fill = strlit_color, color = strlit_color))
    return plot_theme




def InitalizeClassDF(class_path, loc):
    if os.path.exists(class_path):
        class_df = pd.read_csv(class_path)
    else:
        class_df = pd.DataFrame(loc).drop(['geometry'], axis = 1)
        class_df['Class'] = np.nan
        class_df['SubClass'] = np.nan
        class_df.to_csv(class_path, index = False)
        
    return class_df

def UpdateClassDF(loc_id, Class, SubClass, class_path,  new_class, new_subclass):
    if Class == 'Input new':
        st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'Class'] = new_class
    else:
        st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'Class'] = Class
        
        
    if SubClass == 'Input new':
        st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'SubClass'] = new_subclass
    else:
        st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'SubClass'] = SubClass
        
    st.session_state.class_df.to_csv(class_path, index = False)
    
    
# %%
    
def gis_longitude_to_utm_zone(lon):
    """
    Parameters
    ----------
    lon : float
        Longitude.

    Returns
    -------
    utm_zone : int
        UTM zone.

    """
    utm_zone = (math.floor((lon + 180)/6) % 60) + 1
    return utm_zone



def gis_utm_zone_to_proj4(utm_zone):
    """
    Convert UTM Zone to Proj 4 string

    Parameters
    ----------
    utm_zone : Int
        UTM zone as numeric.

    Returns
    -------
    Str
        Proj 4 string for the UTM crs.

    """
    proj4_base = "+proj=utm +zone=UTM_ZONE +datum=WGS84 +units=m +no_defs"
    return proj4_base.replace("UTM_ZONE",str(utm_zone))

# %%

def shift_points_m(pts_gpd, xshift_m, yshift_m):
    """Shift points by x, y meters
    

    Parameters
    ----------
    pts_gpd : geopandas points DataFrame
        Points.
    xshift_m : float or int
        Distance to shift points in meters.
    yshift_m : float or in
        Distance to shift points in meters.

    Returns
    -------
    pts_shifted : geopandas pointsDataFrame
        DataFrame with points shifted.

    """
    # xshift_m = 10
    # yshift_m = 10
    # pts_gpd = gdf

    orig_crs = pts_gpd.crs
    
    
    # get longitude
    pts_4326 = pts_gpd.to_crs(4326)
    pts_lon = pts_4326.Longitude.mean()
    
    # get proj4 string for UTM
    proj4str = gis_utm_zone_to_proj4(gis_longitude_to_utm_zone(pts_lon))
    
    # convert to UTM
    pts_utm_orig = pts_gpd.to_crs(proj4str)
    
    # get x, y to columns
    pts_utm_orig['x'] = pts_utm_orig.geometry.x
    pts_utm_orig['y'] = pts_utm_orig.geometry.y
    
    # drop geometry
    pts_utm_update = pd.DataFrame(pts_utm_orig).drop('geometry', axis = 1)
    
    # adjust x, y columns
    pts_utm_update['xnew'] = pts_utm_update.x + xshift_m
    pts_utm_update['ynew'] = pts_utm_update.y + yshift_m
    
    # convert to geopandas with UTM coordinates
    pts_shifted_utm = gpd.GeoDataFrame(pts_utm_update,  
                                         geometry = gpd.points_from_xy(pts_utm_update.xnew, pts_utm_update.ynew),
                                         crs = proj4str)
    # transform to original crs
    pts_shifted = pts_shifted_utm.to_crs(orig_crs)
    
    return pts_shifted