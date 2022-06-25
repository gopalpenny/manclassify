#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:52:27 2022

@author: gopal
"""

import geopandas as gpd
import pandas as pd
import math
import re

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

test = ['Any', 'Non-farm', 'Farm', '-', 'Not sure']

test in 'Farm'