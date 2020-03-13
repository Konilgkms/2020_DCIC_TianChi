import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
import catboost as cab
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, kurtosis
import datetime

train = pd.read_csv('../data/train0221.csv')
test = pd.read_csv('../data/test0221.csv')
testB = pd.read_csv('../data/testB0221.csv')

def group_feature(df, key, target, aggs):
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t


def extract_feature(df, train):
    t = df.groupby('ID')['speed'].agg({
        'speed_q1': lambda x: np.quantile(x, q=0.25),
        'speed_q3': lambda x: np.quantile(x, q=0.75),
        'speed_kurt': kurtosis
    })
    train = pd.merge(train, t, on='ID', how='left')

    train['speed_iqr'] = train['speed_q3'] - train['speed_q1']

    t = df.groupby('ID')['direction'].agg({
        'direction_q1': lambda x: np.quantile(x, q=0.25),
        'direction_q3': lambda x: np.quantile(x, q=0.75),
        'direction_kurt': kurtosis
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['x'].agg({
        'x_q1': lambda x: np.quantile(x, q=0.25),
        'x_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['y'].agg({
        'y_q1': lambda x: np.quantile(x, q=0.25),
        'y_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['L1_dist'].agg({
        'L1_q1': lambda x: np.quantile(x, q=0.25),
        'L1_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['L2_dist'].agg({
        'L2_q1': lambda x: np.quantile(x, q=0.25),
        'L2_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'x', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'y', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'speed', ['max', 'min', 'mean', 'std', 'skew', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'direction', ['max', 'mean', 'std', 'skew', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'x_i', ['max', 'min', 'mean', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'y_j', ['max', 'min', 'mean', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'L1_dist', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'L2_dist', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'record', ['count'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'total_hour', ['max', 'min'])
    train = pd.merge(train, t, on='ID', how='left')

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']

    train['speed_mean_iqr'] = train['speed_mean'] / train['speed_iqr']

    train['x_i_max_min'] = train['x_i_max'] - train['x_i_min']
    train['y_j_max_min'] = train['y_j_max'] - train['y_j_min']
    train['x_i_max_y_j_min'] = train['x_i_max'] - train['y_j_min']
    train['y_j_max_x_i_min'] = train['y_j_max'] - train['x_i_min']

    train['L1_max_min'] = train['L1_dist_max'] - train['L1_dist_min']
    train['L2_max_min'] = train['L2_dist_max'] - train['L2_dist_min']

    t = group_feature(df, 'ID', 'start_x', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')
    t = group_feature(df, 'ID', 'start_y', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')
    t = group_feature(df, 'ID', 'end_x', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')
    t = group_feature(df, 'ID', 'end_y', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')

    train['end_start_x'] = train['end_x_mean'] - train['start_x_mean']
    train['end_start_y'] = train['end_y_mean'] - train['start_y_mean']
    # train['end_x_start_y'] = train['end_x_mean'] - train['start_y_mean']
    # train['end_y_start_x'] = train['end_y_mean'] - train['start_x_mean']
    train['end_start_L1dis'] = np.abs(train['end_start_x']) + np.abs(train['end_start_y'])
    train['end_start_L2dis'] = np.sqrt(np.square(train['end_start_x']) + np.square(train['end_start_y']))

    # train['end_start_L21_ratio'] = train['end_start_L2dis'] / train['end_start_L1dis']
    train['end_start_L12_ratio'] = train['end_start_L1dis'] / train['end_start_L2dis']

    # train['end_start_slope'] = train['end_start_y'] / np.where(train['end_start_x']==0, 0.001, train['end_start_x'])
    # train['end_start_area'] = train['end_start_x'] * train['end_start_y']

    train['total_hour_max_min'] = train['total_hour_max'] - train['total_hour_min']

    mode_x = df.groupby('ID')['x'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_x'] = train['ID'].map(mode_x)

    mode_y = df.groupby('ID')['y'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_y'] = train['ID'].map(mode_y)

    mode_speed = df.groupby('ID')['speed'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_speed'] = train['ID'].map(mode_speed)

    mode_direction = df.groupby('ID')['direction'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_direction'] = train['ID'].map(mode_direction)

    mode_total_hour = df.groupby('ID')['total_hour'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_total_hour'] = train['ID'].map(mode_total_hour)

    mode_x_i = df.groupby('ID')['x_i'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_x_i'] = train['ID'].map(mode_x_i)

    mode_y_j = df.groupby('ID')['y_j'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_y_j'] = train['ID'].map(mode_y_j)

    mode_L1 = df.groupby('ID')['L1_dist'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_L1'] = train['ID'].map(mode_L1)

    mode_L2 = df.groupby('ID')['L2_dist'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_L2'] = train['ID'].map(mode_L2)

    t = group_feature(df, 'ID', 'hour', ['max', 'min'])
    train = pd.merge(train, t, on='ID', how='left')

    train['distance01'] = train['total_hour_max_min'] * train['speed_mean']
    train['distance02'] = np.sqrt(np.square(train['x_max_x_min']) + np.square(train['y_max_y_min']))
    train['dist01_02_ratio'] = train['distance01'] / train['distance02']

    # train['dist_ratio_01'] = train['end_start_L1dis'] / train['distance01']
    train['dist_ratio_021'] = train['end_start_L2dis'] / train['distance01']
    train['dist_ratio_012'] = train['end_start_L1dis'] / train['distance02']
    train['dist_ratio_02'] = train['end_start_L2dis'] / train['distance02']

    # hour_nunique = df.groupby('ID')['hour'].nunique().to_dict()
    # train['hour_nunique'] = train['ID'].map(hour_nunique)

    x_nunique = df.groupby('ID')['x'].nunique().to_dict()
    train['x_nunique'] = train['ID'].map(x_nunique)

    y_nunique = df.groupby('ID')['y'].nunique().to_dict()
    train['y_nunique'] = train['ID'].map(y_nunique)

    speed_nunique = df.groupby('ID')['speed'].nunique().to_dict()
    train['speed_nunique'] = train['ID'].map(speed_nunique)

    direction_nunique = df.groupby('ID')['direction'].nunique().to_dict()
    train['direction_nunique'] = train['ID'].map(direction_nunique)

    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])

    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    return train

train_label = train.drop_duplicates('ID')
test_label = test.drop_duplicates('ID')
testB_label = testB.drop_duplicates('ID')

train_label['type'].value_counts(1)

type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)

train_label = extract_feature(train, train_label)
test_label = extract_feature(test, test_label)
testB_label = extract_feature(testB, testB_label)

features = [x for x in train_label.columns if x not in ['ID','type','total_hour_max','total_hour_min',
                                                        'time','month','weekday','day','hour','minute',
                                                        'end_start_x_mean','end_start_y_mean']]
target = 'type'

print(len(features), ','.join(features))

X = train_label[features].copy()
y = train_label[target]

llf = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='multiclass',
    n_estimators=5000,
    metrics='multi_error',
    lambda_l1=1e-4,
    lambda_l2=1e-4,
    min_child_samples=50,
    bagging_fraction=0.6,
    num_class=3,
)
'''
xlf = xgb.XGBClassifier(
    n_estimators=5000,
    objective='multi:softprob',
    reg_alpha=1e-4,
    reg_lambda=1e-4,
    min_child_samples=50,
    bagging_fraction=0.6,
    num_class=3,
)

clf = cab.CatBoostClassifier(
    iterations=20000,
    loss_function='MultiClass',
    l2_leaf_reg=1e-4,
    logging_level='Verbose',
    classes_count=3,
)
'''
'''
xgb_pred = np.zeros((len(testB_label), 3))
xgb_oof = np.zeros((len(X), 3))
sk = StratifiedKFold(n_splits=8, shuffle=True, random_state=2019)
for index, (train, test) in enumerate(sk.split(X, y)):
    x_train = X.iloc[train]
    y_train = y.iloc[train]
    x_test = X.iloc[test]
    y_test = y.iloc[test]
    xlf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='merror', early_stopping_rounds=1000,
            verbose=500)
    pred_xgb = xlf.predict_proba(x_test)
    xgb_oof[test] = pred_xgb
    pred_xgb = np.argmax(pred_xgb, axis=1)
    weight_xgb = metrics.f1_score(y_test, pred_xgb, average='macro')
    print(index, 'val f1', weight_xgb)

    test_pred = xlf.predict_proba(testB_label[features])
    xgb_pred += test_pred / 8

xgb_oof_1 = np.argmax(xgb_oof, axis=1)
print('oof f1', metrics.f1_score(xgb_oof_1, y, average='macro'))

cab_pred = np.zeros((len(testB_label), 3))
cab_oof = np.zeros((len(X), 3))
sk = StratifiedKFold(n_splits=8, shuffle=True, random_state=2019)
for index, (train, test) in enumerate(sk.split(X, y)):
    x_train = X.iloc[train]
    y_train = y.iloc[train]
    x_test = X.iloc[test]
    y_test = y.iloc[test]
    clf.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=2000, verbose=500)
    pred_cab = clf.predict_proba(x_test)
    cab_oof[test] = pred_cab
    pred_cab = np.argmax(pred_cab, axis=1)
    weight_cab = metrics.f1_score(y_test, pred_cab, average='macro')
    print(index, 'val f1', weight_cab)

    test_pred = clf.predict_proba(testB_label[features])
    cab_pred += test_pred / 8

cab_oof_1 = np.argmax(cab_oof, axis=1)
print('oof f1', metrics.f1_score(cab_oof_1, y, average='macro'))
'''
lgb_pred = np.zeros((len(testB_label), 3))
lgb_oof = np.zeros((len(X), 3))
sk = StratifiedKFold(n_splits=8, shuffle=True, random_state=2019)
for index, (train, test) in enumerate(sk.split(X, y)):
    x_train = X.iloc[train]
    y_train = y.iloc[train]
    x_test = X.iloc[test]
    y_test = y.iloc[test]
    llf.fit(x_train, y_train, eval_set=(x_test, y_test), eval_metric='multi_error', early_stopping_rounds=1000,
            verbose=500)
    pred_lgb = llf.predict_proba(x_test)
    lgb_oof[test] = pred_lgb
    pred_lgb = np.argmax(pred_lgb, axis=1)
    weight_lgb = metrics.f1_score(y_test, pred_lgb, average='macro')
    print(index, 'val f1', weight_lgb)

    test_pred = llf.predict_proba(testB_label[features])
    lgb_pred += test_pred / 8

lgb_oof_1 = np.argmax(lgb_oof, axis=1)
print('oof f1', metrics.f1_score(lgb_oof_1, y, average='macro'))
'''
oof = 0.55*lgb_oof + 0.3*xgb_oof + 0.15*cab_oof
oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))

pred = np.zeros((len(test_label),3))
pred = 0.3*xgb_pred + 0.55*lgb_pred +0.15*cab_pred
'''
pred = np.zeros((len(test_label),3))
pred = lgb_pred

pred = np.argmax(pred, axis=1)
sub = testB_label[['ID']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))

sub['ID'] = sub['ID']
sub['pred'] = sub['pred'].map(type_map_rev)
sub.sort_values("ID", inplace=True)
sub.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), index=None, header=None)