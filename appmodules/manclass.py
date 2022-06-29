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
import json
# import ?

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research/Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemod import rs
from geemod import eesentinel as ees
from dateutil.relativedelta import relativedelta
from datetime import datetime
# appmodule

def testfunc():
    st.write("success")
        
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
    
    # Initialize Earth Engine, if necessary
    try:
        pt_dummy = ee.Geometry.Point([100, 10])
    except:
        ee.Initialize()

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
    s2 = s2.assign(NDWI = lambda df: (df.B8 - df.B4)/(df.B8 + df.B4))
    
    
    s2_long = s2.melt(id_vars = ['datetime','cloudmask'], value_vars = ['B8','B4','B3','B2','NDVI'])
    
    return s2_long
    
    # # s2['backscatter'] = (s2['VV']**2 + s2['VH']**2) ** (1/2)

def plotTimeseries(loc_id, timeseries_dir_path, date_range, month_seq, snapshot_dates, spectra_list):
    """
    

    Parameters
    ----------
    loc_id : Int
        Integer of the location ID.
    timeseries_dir_path : Str
        path to the directory containing timeseries data.
    date_range : list
        List of strings in "%Y-%m-%d" format
    month_seq : List
        List of datetime.datetime values indicating the axis ticks and vertical gridlines.
    snapshot_dates : List
        List of datetime.datetime objects indicating the dates of the snapshot images in Classify page.
    spectra_list : list
        List of lists. Each sub-list contains the date ranges for spectra plot in Classify page.

    Returns
    -------
    p_sentinel : matplotlib.plt
        Plot of timeseries for the points in question.

    """
    
    s1 = GenS1data(loc_id, timeseries_dir_path, date_range)
    s2 = GenS2data(loc_id, timeseries_dir_path, date_range)
    s2 = s2[s2['variable'] == 'NDVI']
    
    datetime_range = [datetime.strptime(x, '%Y-%m-%d') for x in date_range]
    
    year_begin_datetime = datetime.strftime(datetime_range[len(datetime_range)-1],"%Y-01-01")
    
    sentinel = pd.concat([s1, s2])
    
    start_date = datetime_range[0]
    end_date = datetime_range[1]
    pre_date = start_date - relativedelta(months = 6)
    post_date = end_date + relativedelta(months = 6)
    
    month_seq_df = pd.DataFrame({'datetime':month_seq})

    line_vars = ['NDVI']
    smooth_vars = ['backscatter']
    sentinel_cloudfree = sentinel.query('cloudmask != 1')
    sentinel_cloudfree['date'] = [datetime.strftime(x, '%Y-%m-%d') for x in sentinel_cloudfree['datetime']]
    
    # snapshots
    snapshot_datetimes = pd.DataFrame({
        'date' : snapshot_dates,
        'snapshot' : [str(x) for x in list(range(len(snapshot_dates)))]})
    sentinel_snapshots = sentinel_cloudfree.merge(snapshot_datetimes, how = 'inner', on = 'date')

    # Build date range for spectra
    spectral_range = pd.DataFrame(columns = ['id','start_date', 'end_date', 'variable'])
    for i in range(len(spectra_list)):
        row_list = [i] + list(spectra_list[i]) + [line_vars[0]]
        spectral_range.loc[i] = row_list
        
    var_min = sentinel_cloudfree[sentinel_cloudfree['variable'] == line_vars[0]].value.min()
    var_max = sentinel_cloudfree[sentinel_cloudfree['variable'] == line_vars[0]].value.max()
    spectral_range['yval'] = var_min + spectral_range.id * (var_max - var_min) * 0.05
    
    month_seq_labels_full = [datetime.strftime(x, "%b %d, %Y") for x in month_seq]
    month_seq_labels_brief = [datetime.strftime(x, "%b %d") for x in month_seq]
    month_seq_labels = [month_seq_labels_full[i] if i == 0 or i == (len(month_seq)-len(month_seq)) else month_seq_labels_brief[i] for i in range(len(month_seq))]
    
    p_sentinel = (
        p9.ggplot(data = sentinel_cloudfree, mapping = p9.aes('datetime', 'value')) + 
        p9.annotate('rect',xmin = start_date, xmax = end_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'white', color = 'black', alpha = 1) +
        p9.geom_segment(data = spectral_range, mapping = p9.aes(x = 'start_date', xend = 'end_date', y = 'yval', yend = 'yval',color = 'id'), size = 2) +
        p9.geom_vline(data = month_seq_df, mapping = p9.aes(xintercept = 'datetime'), color = 'black', alpha = 0.5) +
        p9.annotate('vline', xintercept = year_begin_datetime, color = 'gray', linetype = 'dashed', alpha = 0.5) +
        # p9.annotate('rect',xmin = end_date, xmax = post_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'black', alpha = 0.5) +
        p9.geom_point() + 
        p9.geom_line(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(line_vars)]) + 
        p9.geom_point(data = sentinel_snapshots, mapping=p9.aes(fill = 'snapshot'), size = 4) + 
        p9.geom_smooth(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(smooth_vars)], span = 0.25) + 
        p9.facet_wrap('variable', scales = 'free_y',ncol = 1) +
        # p9.xlim()+
        # p9.scale_x_datetime(limits = [datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)], 
        p9.scale_x_datetime(limits = [pre_date, post_date], breaks = month_seq, labels = month_seq_labels) + # date_labels = '%Y-%b', date_breaks = '1 year') +
        PlotTheme() + p9.theme(axis_title_x = p9.element_blank(),
                               panel_background= p9.element_rect(fill = 'gray', color = None),
                               # panel_border= p9.element_rect(fill = 'gray'),
                               legend_position = 'none'))
    
    return p_sentinel

def plotSpectra(loc_id, timeseries_dir_path, date_range, spectra_list):
    
    s1 = GenS1data(loc_id, timeseries_dir_path, date_range)
    s2 = GenS2data(loc_id, timeseries_dir_path, date_range)
    sentinel = pd.concat([s1, s2]).query('cloudmask != 1')
    # sentinel['date'] = [datetime.strftime(x, '%Y-%m-%d') for x in sentinel['datetime']]
    
    s2_bands = pd.DataFrame({
        'variable' : ['B2','B3','B4','B8','backscatter','NDVI'],
        'freq_nm': [495, 560, 660, 835, -1, -1]
        })
    
    sent_range_all = pd.DataFrame(columns = list(sentinel.columns) + ['rangenum'])
    for i in range(len(spectra_list)):
        range_prep = sentinel.loc[(spectra_list[i][0] <= sentinel['datetime']) & (sentinel['datetime'] <= spectra_list[i][1])]
        range_prep['rangenum'] = i
        sent_range_all = sent_range_all.append(range_prep)
        
    sent_range_all = sent_range_all.merge(s2_bands, how = 'left', on = 'variable')
    
    sent_range = (sent_range_all
      .groupby(['variable','rangenum','freq_nm'], as_index = False)
      .agg({'value' : 'mean'}))
    
    sent_range['Reflectance'] = sent_range['value'] / 1e4
    
    print(sent_range)
    strlit_color = '#0F1116'
    p_spectra = (
        p9.ggplot(data = sent_range[sent_range['freq_nm'] > 0], mapping = p9.aes(x = 'freq_nm', y = 'Reflectance', color = 'rangenum')) +
        p9.geom_line(size = 2) + p9.geom_point(size = 5) + 
        p9.xlab('Frequency, nm') +
        p9.expand_limits(y = (0, 0.2)) +
        p9.scale_color_manual(['#572851','#FAE855']) +
        PlotTheme() +
        p9.theme(legend_position = 'none',
                 panel_background = p9.element_rect(fill = strlit_color, color = 'lightgray'),
                  panel_grid_major = p9.element_line(color = 'lightgray', size = 0.5),
                 figure_size = (4.5,2)))
    
    return p_spectra
    
    # line_vars = ['NDVI']
    # bar_vars = ['backscatter']
    # sentinel_cloudfree = sentinel.query('cloudmask != 1')
    
    # # snapshots
    # snapshot_datetimes = pd.DataFrame({
    #     'date' : snapshot_dates,
    #     'snapshot' : [str(x) for x in list(range(len(snapshot_dates)))]})
    # sentinel_snapshots = sentinel_cloudfree.merge(snapshot_datetimes, how = 'inner', on = 'date')
    
    # # date range for spectra
    # spectral_range = pd.DataFrame(columns = ['id','start_date', 'end_date', 'variable'])
    # for i in range(len(spectra_list)):
    #     row_list = [i] + list(spectra_list[i]) + [line_vars[0]]
    #     spectral_range.loc[i] = row_list
        
        
    # var_min = sentinel_cloudfree[sentinel_cloudfree['variable'] == line_vars[0]].value.min()
    # print(var_min)
    # # .value.min()
    # var_max = sentinel_cloudfree[sentinel_cloudfree['variable'] == line_vars[0]].value.max()
    # # print('var_min')
    # # print(type(var_min))
    # # print(type(spectral_range.id))
    # spectral_range['yval'] = var_min + spectral_range.id * (var_max - var_min) * 0.05
        
    # print(spectral_range)
    # # spectral_range
    # # print(sentinel_cloudfree.variable[0])
    
    # p_sentinel = (
    #     p9.ggplot(data = sentinel_cloudfree, mapping = p9.aes('datetime', 'value')) + 
    #     p9.annotate('rect',xmin = start_date, xmax = end_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'white', color = 'black', alpha = 1) +
    #     p9.geom_segment(data = spectral_range, mapping = p9.aes(x = 'start_date', xend = 'end_date', y = 'yval', yend = 'yval',color = 'id'), size = 2) +
    #     p9.geom_vline(data = month_seq_df, mapping = p9.aes(xintercept = 'datetime'), color = 'black', alpha = 0.5) +
    #     # p9.annotate('rect',xmin = end_date, xmax = post_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'black', alpha = 0.5) +
    #     p9.geom_point() + 
    #     p9.geom_line(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(line_vars)]) + 
    #     p9.geom_point(data = sentinel_snapshots, mapping=p9.aes(fill = 'snapshot'), size = 4) + 
    #     p9.geom_smooth(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(smooth_vars)], span = 0.25) + 
    #     p9.facet_wrap('variable', scales = 'free_y',ncol = 1) +
    #     # p9.xlim()+
    #     # p9.scale_x_datetime(limits = [datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)], 
    #     p9.scale_x_datetime(limits = [pre_date, post_date], 
    #                         date_labels = '%Y-%b', date_breaks = '1 year') +
    #     PlotTheme() + p9.theme(axis_title_x = p9.element_blank(),
    #                            panel_background= p9.element_rect(fill = 'gray', color = None),
    #                            # panel_border= p9.element_rect(fill = 'gray'),
    #                            legend_position = 'none'))
    
    # return p_sentinel


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




# %%
    

# %%