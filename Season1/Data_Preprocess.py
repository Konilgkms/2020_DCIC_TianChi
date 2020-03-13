import numpy as np
import pandas as pd
import datetime

#----read data----#
train = pd.read_csv("../data/train_origin_0221.csv")
test = pd.read_csv("../data/test_origin_0221.csv")
testB = pd.read_csv("../data/testB_origin_0221.csv")

#----rename columns----#
train.rename(columns={'渔船ID':'ID', '速度':'speed', '方向':'direction'}, inplace=True)
test.rename(columns={'渔船ID':'ID', '速度':'speed', '方向':'direction'}, inplace=True)
testB.rename(columns={'渔船ID':'ID', '速度':'speed', '方向':'direction'}, inplace=True)

#----data preprocess----#
train_df = train[(train['speed'] > 0)]
train_df = train_df[(train_df['speed'] <= 10)]
train_df['record'] = 1
train_df['time'] = train_df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
train_df['month'] = train_df['time'].dt.month
train_df['day'] = train_df['time'].dt.day
train_df['hour'] = train_df['time'].dt.hour
train_df['minute'] = train_df['time'].dt.minute
train_df['weekday'] = train_df['time'].dt.weekday
train_df['total_hour'] = (train_df['month'].astype(float) - 10)*31*24 + train_df['day'].astype(float)*24 + train_df['hour'].astype(float) + train_df['minute'].astype(float) / 60
train_df['hour_minute'] = train_df['hour']*60 + train_df['minute']
#----处理数据的时候发现有些ship的x,y一直未变，且x均为6165599.368591921，y均为5202659.922186158，猜测是港口位置----#
train_df['x_i'] = train_df['x'] - 6165599.368591921
train_df['y_j'] = train_df['y'] - 5202659.922186158
train_df['L1_dist'] = np.abs(train_df['x'] - 6165599.368591921) + np.abs(train_df['y'] - 5202659.922186158)
train_df['L2_dist'] =np.sqrt(np.square(train_df['x_i']) + np.square(train_df['y_j']))
train_df.to_csv('../data/train0221.csv', index = None)

test_df = test[(test['speed'] > 0)]
test_df = test_df[(test_df['speed'] <= 10)]
test_df['record'] = 1
test_df['time'] = test_df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
test_df['month'] = test_df['time'].dt.month
test_df['day'] = test_df['time'].dt.day
test_df['hour'] = test_df['time'].dt.hour
test_df['minute'] = test_df['time'].dt.minute
test_df['weekday'] = test_df['time'].dt.weekday
test_df['total_hour'] = (test_df['month'].astype(float) - 10)*31*24 + test_df['day'].astype(float)*24 + test_df['hour'].astype(float) + test_df['minute'].astype(float) / 60
test_df['hour_minute'] = test_df['hour']*60 + test_df['minute']
test_df['x_i'] = test_df['x'] - 6165599.368591921
test_df['y_j'] = test_df['y'] - 5202659.922186158
test_df['L1_dist'] = np.abs(test_df['x'] - 6165599.368591921) + np.abs(test_df['y'] - 5202659.922186158)
test_df['L2_dist'] =np.sqrt(np.square(test_df['x_i']) + np.square(test_df['y_j']))
test_df.to_csv('../data/test0221.csv', index = None)

testB_df = testB[(testB['speed'] > 0)]
testB_df = testB_df[(testB_df['speed'] <= 10)]
testB_df['record'] = 1
testB_df['time'] = testB_df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
testB_df['month'] = testB_df['time'].dt.month
testB_df['day'] = testB_df['time'].dt.day
testB_df['hour'] = testB_df['time'].dt.hour
testB_df['minute'] = testB_df['time'].dt.minute
testB_df['weekday'] = testB_df['time'].dt.weekday
testB_df['total_hour'] = (testB_df['month'].astype(float) - 10)*31*24 + testB_df['day'].astype(float)*24 + testB_df['hour'].astype(float) + testB_df['minute'].astype(float) / 60
testB_df['hour_minute'] = testB_df['hour']*60 + testB_df['minute']
testB_df['x_i'] = testB_df['x'] - 6165599.368591921
testB_df['y_j'] = testB_df['y'] - 5202659.922186158
testB_df['L1_dist'] = np.abs(testB_df['x'] - 6165599.368591921) + np.abs(testB_df['y'] - 5202659.922186158)
testB_df['L2_dist'] =np.sqrt(np.square(testB_df['x_i']) + np.square(testB_df['y_j']))

testB_df.to_csv('../data/testB0221.csv', index = None)