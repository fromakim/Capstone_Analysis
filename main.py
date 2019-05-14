# %% In[0]: Basic Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
import cv2
import os
import re
import csv

data_overall = pd.DataFrame()
soft_grip = True                            # HACK: Toggle if want to check hard-grip

# %% In[1]: Get Response Data
res = pd.read_excel('./data/응답.xlsx', sheet_name = 'RPE V2')

# HACK: If hard grip is in concern, change 2 -> 4
# Set Label Data
data_overall = res.loc[list(range(2 if soft_grip else 4, res.shape[0], 6)), ['overall']].reset_index().loc[:, ['overall']].rename(columns = {'overall': 'label'})

# Set Prediction and Explore Response
data_overall['prediction'] = res.loc[list(range(0, res.shape[0], 6)), ['overall']].reset_index().loc[:, ['overall']]
data_overall['explore'] = res.loc[list(range(1, res.shape[0], 6)), ['overall']].reset_index().loc[:, ['overall']]

# %% In[2]: Get Raw Center of Pressure Data
subjects = os.listdir('./data/텍스캔/')
subjects.sort()

# Center of Force Data: Use middle 2 second only
texcanFormat = re.compile('\w*[0-9]+R_C.csv')

cfd = pd.DataFrame(columns = ['CF_row_mean', 'CF_row_std', 'CF_col_mean', 'CF_col_std', 'CF_rowsum_mean', 'CF_rowsum_std'])
for sub in subjects:
    tex = [s for s in os.listdir('./data/텍스캔/%s' % sub) if texcanFormat.match(s)]
    tex.sort()
    tex = tex[:-5]
    tex = tex[0 if soft_grip else 1:len(tex):2]

    for texdata in tex:
        ps = pd.DataFrame(columns = ['Frame', 'Time', 'Absolute Time', 'Row', 'Col', 'Row Sum'])

        with open('./data/텍스캔/%s/%s' % (sub, texdata), newline='', encoding='euc-kr') as texfile:
            reader = csv.reader(texfile)
            for r in reader:
                if len(r) == 6 and r[0] != 'COMMENTS Frame':
                    row = [float(r[0]), float(r[1]), r[2], float(r[3]), float(r[4]), float(r[5])]
                    ps.loc[len(ps)] = row

        ps = ps[51:151]

        cfd.loc[len(cfd)] = [ps['Row'].mean(), ps['Row'].std(), ps['Col'].mean(), ps['Col'].std(), ps['Row Sum'].mean(), ps['Row Sum'].std()]
data_overall = data_overall.join(cfd)

# %% In[3]: Get Normalized Regional Pressure Data
for sub in subjects:
    tex = [s for s in os.listdir('./data/텍스캔/%s' % sub) if texcanFormat.match(s)]
    tex.sort()

    # MVPs
    mvp = tex[-5:]

    for m in mvp:
        with open('./data/텍스캔/%s/%s' % (sub, texdata), newline='', encoding='euc-kr') as texfile:
            reader = csv.reader(texfile)
            for r in reader:
                if len(r) == 25 and r[0] != 'COMMENTS Frame':
                    row = [float(r[0]), float(r[1]), r[2], float(r[3]), float(r[4]), float(r[5])]
                    ps.loc[len(ps)] = row

    # Regional Pressure
    tex = tex[:-5]
