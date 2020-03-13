import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis
import datetime

train_dir = ""
test_dir = ""

test = pd.DataFrame(columns=['渔船ID', 'lat', 'lon', '速度', '方向', 'time', 'start_lat', 'start_lon', 'end_lat',
                             'end_lon', 'lat_diff', 'lon_diff', 'v_diff', 'dist_diff', 'lat_lon'])

for root, dirs, files in os.walk(test_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)

        size = temp.shape[0]

        temp['lat'] = temp['lat'].round(3)
        temp['lon'] = temp['lon'].round(3)

        start_lat = temp['lat'][size - 1]
        start_lon = temp['lon'][size - 1]
        temp['start_lat'] = start_lat
        temp['start_lon'] = start_lon
        end_lat = temp['lat'][0]
        end_lon = temp['lon'][0]
        temp['end_lat'] = end_lat
        temp['end_lon'] = end_lon

        temp['lat_last'] = temp['lat'].shift(1)
        temp['lat_last'][0] = temp['lat'][0] + 0.001
        temp['lat_diff'] = temp['lat'] - temp['lat_last']

        temp['lon_last'] = temp['lon'].shift(1)
        temp['lon_last'][0] = temp['lon'][0] + 0.001
        temp['lon_diff'] = temp['lon'] - temp['lon_last']

        temp['v_last'] = temp['速度'].shift(1)
        temp['v_last'][0] = temp['速度'][0] + 0.001
        temp['v_diff'] = temp['速度'] - temp['v_last']

        temp['dist_diff'] = 6371 * np.arccos(np.sin(temp['lat'] * np.pi / 180)*np.sin(temp['lat_last'] * np.pi / 180) +
                                             np.cos(temp['lat'] * np.pi / 180)*np.cos(temp['lat_last'] * np.pi / 180) *
                                             np.cos((temp['lon'] - temp['lon_last']) * np.pi / 180))

        temp['dist_diff'] = temp['dist_diff'].round(4)

        temp['lat_lon'] = temp['lat'].cov(temp['lon'])

        del temp['lat_last'], temp['lon_last'], temp['v_last']

        #temp['速度'] = temp['dist_diff'] * 6 / 1.852

        test = pd.concat([test, temp])
print(test.info())
test.to_csv("./test_origin.csv", index=False)

train = pd.DataFrame(columns=['渔船ID', 'lat', 'lon', '速度', '方向', 'time', 'type', 'start_lat', 'start_lon',
                              'end_lat', 'end_lon', 'lat_diff', 'lon_diff', 'v_diff', 'dist_diff', 'lat_lon',
                              ])

for root, dirs, files in os.walk(train_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)

        size = temp.shape[0]

        temp['lat'] = temp['lat'].round(3)
        temp['lon'] = temp['lon'].round(3)

        start_lat = temp['lat'][size - 1]
        start_lon = temp['lon'][size - 1]
        temp['start_lat'] = start_lat
        temp['start_lon'] = start_lon
        end_lat = temp['lat'][0]
        end_lon = temp['lon'][0]
        temp['end_lat'] = end_lat
        temp['end_lon'] = end_lon

        temp['lat_last'] = temp['lat'].shift(1)
        temp['lat_last'][0] = temp['lat'][0] + 0.001
        temp['lat_diff'] = temp['lat'] - temp['lat_last']

        temp['lon_last'] = temp['lon'].shift(1)
        temp['lon_last'][0] = temp['lon'][0] + 0.001
        temp['lon_diff'] = temp['lon'] - temp['lon_last']

        temp['v_last'] = temp['速度'].shift(1)
        temp['v_last'][0] = temp['速度'][0] + 0.001
        temp['v_diff'] = temp['速度'] - temp['v_last']

        temp['dist_diff'] = 6371 * np.arccos(
            np.sin(temp['lat'] * np.pi / 180) * np.sin(temp['lat_last'] * np.pi / 180) +
            np.cos(temp['lat'] * np.pi / 180) * np.cos(temp['lat_last'] * np.pi / 180) *
            np.cos((temp['lon'] - temp['lon_last']) * np.pi / 180))

        temp['dist_diff'] = temp['dist_diff'].round(4)

        temp['lat_lon'] = temp['lat'].cov(temp['lon'])

        del temp['lat_last'], temp['lon_last'], temp['v_last']

        #temp['速度'] = temp['dist_diff'] * 6 / 1.852

        train = pd.concat([train, temp])
print(train.info())
train.to_csv("./train_origin.csv", index=False)
