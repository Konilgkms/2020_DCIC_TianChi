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

#----read data----#
train = pd.read_csv("./train_origin.csv")
test = pd.read_csv("./test_origin.csv")


#----rename columns----#
train.rename(columns={'渔船ID':'ID', '速度':'speed', '方向':'direction'}, inplace=True)
test.rename(columns={'渔船ID':'ID', '速度':'speed', '方向':'direction'}, inplace=True)

#----data preprocess----#
train_df = train[(train['speed'] >= 0)]
train_df = train_df[(train_df['speed'] <= 10)]
train_df['record'] = 0
train_df.loc[train_df.lat_diff > 0.001,'record'] = 1
train_df.loc[train_df.lon_diff > 0.001,'record'] = 1
train_df['time'] = train_df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
train_df['month'] = train_df['time'].dt.month
train_df['day'] = train_df['time'].dt.day
train_df['hour'] = train_df['time'].dt.hour
train_df['minute'] = train_df['time'].dt.minute
train_df['weekday'] = train_df['time'].dt.weekday
train_df['total_hour'] = (train_df['month'].astype(float) - 7)*31*24 + train_df['day'].astype(float)*24 + train_df['hour'].astype(float) + train_df['minute'].astype(float) / 60
train_df['hour_minute'] = train_df['hour']*60 + train_df['minute']

train_df['lat_j_1'] = train_df['lat'] - 24.392
train_df['lon_i_1'] = train_df['lon'] - 119.205

train_df['lat_j_2'] = train_df['lat'] - 26.046
train_df['lon_i_2'] = train_df['lon'] - 119.597

train_df['dist_gap_1'] = 6371 * np.arccos(np.sin(train_df['lat'] * np.pi / 180)*np.sin(24.392 * np.pi / 180) +
                                          np.cos(train_df['lat'] * np.pi / 180)*np.cos(24.392 * np.pi / 180) *
                                          np.cos((train_df['lon'] - 119.205) * np.pi / 180))

train_df['dist_gap_2'] = 6371 * np.arccos(np.sin(train_df['lat'] * np.pi / 180)*np.sin(26.046 * np.pi / 180) +
                                          np.cos(train_df['lat'] * np.pi / 180)*np.cos(26.046 * np.pi / 180) *
                                          np.cos((train_df['lon'] - 119.597) * np.pi / 180))


train_df['lat_minus_mean'] = train_df['lat'].values - train_df.groupby('ID')['lat'].transform('mean').values
train_df['lon_minus_mean'] = train_df['lon'].values - train_df.groupby('ID')['lon'].transform('mean').values
train_df['lat_minus_mean_std'] = train_df['lat_minus_mean'] / train_df.groupby('ID')['lat'].transform('std').values
train_df['lon_minus_mean_std'] = train_df['lon_minus_mean'] / train_df.groupby('ID')['lon'].transform('std').values

train_df.to_csv('./train_concat.csv', index = None)

test_df = test[(test['speed'] >= 0)]
test_df = test_df[(test_df['speed'] <= 10)]
test_df['record'] = 0
test_df.loc[test_df.lat_diff > 0.001,'record'] = 1
test_df.loc[test_df.lon_diff > 0.001,'record'] = 1
test_df['time'] = test_df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
test_df['month'] = test_df['time'].dt.month
test_df['day'] = test_df['time'].dt.day
test_df['hour'] = test_df['time'].dt.hour
test_df['minute'] = test_df['time'].dt.minute
test_df['weekday'] = test_df['time'].dt.weekday
test_df['total_hour'] = (test_df['month'].astype(float) - 7)*31*24 + test_df['day'].astype(float)*24 + test_df['hour'].astype(float) + test_df['minute'].astype(float) / 60
test_df['hour_minute'] = test_df['hour']*60 + test_df['minute']

test_df['lat_j_1'] = test_df['lat'] - 24.392
test_df['lon_i_1'] = test_df['lon'] - 119.205

test_df['dist_gap_1'] = 6371 * np.arccos(np.sin(test_df['lat'] * np.pi / 180)*np.sin(24.392 * np.pi / 180) +
                                         np.cos(test_df['lat'] * np.pi / 180)*np.cos(24.392 * np.pi / 180) *
                                         np.cos((test_df['lon'] - 119.205) * np.pi / 180))

test_df['lat_j_2'] = test_df['lat'] - 26.046
test_df['lon_i_2'] = test_df['lon'] - 119.597

test_df['dist_gap_2'] = 6371 * np.arccos(np.sin(test_df['lat'] * np.pi / 180)*np.sin(26.046 * np.pi / 180) +
                                         np.cos(test_df['lat'] * np.pi / 180)*np.cos(26.046 * np.pi / 180) *
                                         np.cos((test_df['lon'] - 119.597) * np.pi / 180))

test_df['lat_lon'] = test_df['lat'].cov(test_df['lon'])
test_df['lat_minus_mean'] = test_df['lat'].values - test_df.groupby('ID')['lat'].transform('mean').values
test_df['lon_minus_mean'] = test_df['lon'].values - test_df.groupby('ID')['lon'].transform('mean').values
test_df['lat_minus_mean_std'] = test_df['lat_minus_mean'] / test_df.groupby('ID')['lat'].transform('std').values
test_df['lon_minus_mean_std'] = test_df['lon_minus_mean'] / test_df.groupby('ID')['lon'].transform('std').values


test_df.to_csv('./test_concat.csv', index = None)