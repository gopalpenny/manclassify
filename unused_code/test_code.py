#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:52:27 2022

@author: gopal
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotnine as p9

df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
      'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
      'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
      'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})

# %%

df[df.Latitude >= 0.56]

# %%

# [df.City.iloc[i] == 'Brasilia' for i in range(len(df.City))]

idx = df.index[df.City in ['Brasilia']].tolist()[0] + 1

# df.loc[idx]

# %%


newidx = [idx[0] + 1]

# newidx

df.loc[newidx, 'Latitude'] + 1

# %%

# [i for i in range(len(dfSant.City)) if dfSant.City[i] in ['Santiago']]

idx = [i for i in range(len(df.City)) if df.City[i] in ['Brasilia']]
new = df.City.shift(-1)[idx]

new
# new.iloc[0]
# %%

df.City

# %%

filterargs = {
    'lon' : [-180, 180],
    'lat' : [-90, 90],
    'Class' : 'Any',
    'Subclass' : 'Any',
    'Downloaded' : 'Yes'
    }

# %%

filterargs['lon'][0]

# %%
strlit_color = '#0F1116'
darktheme = p9.theme(panel_background = p9.element_rect(fill = None),      
                 panel_border = p9.element_rect(color = None),
                 panel_grid_major=p9.element_blank(),
                 panel_grid_minor=p9.element_blank(),
                 axis_text = p9.element_text(color = 'white'),
                 axis_ticks = p9.element_line(color = 'red'),
                 axis_title = p9.element_text(color = 'white'),
                 # plot_background=p9.element_rect(fill = 'black'),
                 plot_background=p9.element_rect(fill = strlit_color, color = strlit_color))

# %%

start_date = datetime.strptime('2000-06-01', '%Y-%m-%d')


type(pd.to_datetime(start_date))
# start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
end_date = start_date + relativedelta(years = 1)
pre_date = start_date - relativedelta(years = 1)
post_date = end_date + relativedelta(years = 1)

month_increment = 4
length_out = 4
month_seq = pd.DataFrame({'datetime':[start_date + relativedelta(months = x) for x in np.arange(length_out) * month_increment],
                          'val' : np.arange(length_out)})
# def get_month_sequence(start_date, month_increment, length_out):
#     month_seq = pd.DataFrame({'vbars':[start_date + relativedelta(months = x) for x in np.arange(length_out) * month_increment]})
    
#     return(month_seq)


strlit_color = '#0F1116'
darktheme = p9.theme(axis_ticks_length=30,
                     axis_ticks_major_x = p9.element_line(color='red',size = 3))
    
(p9.ggplot() + 
  p9.geom_vline(data = month_seq, mapping = p9.aes(xintercept = 'datetime')) + 
    p9.annotate('rect',xmin = pre_date, xmax = post_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'black', alpha = 0.2) +
   p9.geom_point(data = month_seq, mapping = p9.aes(x = 'datetime', y = 'val')) + darktheme) 


