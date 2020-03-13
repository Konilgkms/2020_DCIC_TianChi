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

train = pd.read_csv('./train_concat.csv')
test = pd.read_csv('./test_concat.csv')

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
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lat'].agg({
        'lat_q1': lambda x: np.quantile(x, q=0.25),
        'lat_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lon'].agg({
        'lon_q1': lambda x: np.quantile(x, q=0.25),
        'lon_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lat_j_1'].agg({
        'lat_j_1_q1': lambda x: np.quantile(x, q=0.25),
        'lat_j_1_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lon_i_1'].agg({
        'lon_i_1_q1': lambda x: np.quantile(x, q=0.25),
        'lon_i_1_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lat_j_2'].agg({
        'lat_j_2_q1': lambda x: np.quantile(x, q=0.25),
        'lat_j_2_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lon_i_2'].agg({
        'lon_i_2_q1': lambda x: np.quantile(x, q=0.25),
        'lon_i_2_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['dist_gap_1'].agg({
        'dist_gap_1_q1': lambda x: np.quantile(x, q=0.25),
        'dist_gap_1_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['dist_gap_2'].agg({
        'dist_gap_2_q1': lambda x: np.quantile(x, q=0.25),
        'dist_gap_2_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['dist_diff'].agg({
        'dist_diff_q1': lambda x: np.quantile(x, q=0.25),
        'dist_diff_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lat_diff'].agg({
        'lat_diff_q1': lambda x: np.quantile(x, q=0.25),
        'lat_diff_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')

    t = df.groupby('ID')['lon_diff'].agg({
        'lon_diff_q1': lambda x: np.quantile(x, q=0.25),
        'lon_diff_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')
    '''
    t = df.groupby('ID')['v_diff'].agg({
        'v_diff_q1': lambda x: np.quantile(x, q=0.25),
        'v_diff_q3': lambda x: np.quantile(x, q=0.75),
    })
    train = pd.merge(train, t, on='ID', how='left')
    '''
    t = group_feature(df, 'ID', 'lat', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lon', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lat_j_1', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lon_i_1', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lat_j_2', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lon_i_2', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'dist_gap_1', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'dist_gap_2', ['max', 'min', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'speed', ['max', 'mean', 'std', 'skew', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'direction', ['max', 'mean', 'std', 'skew', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'record', ['count'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'total_hour', ['max', 'min'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'hour_minute', ['max', 'min'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'dist_diff', ['max', 'mean', 'std', 'skew', 'median', 'sum'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lat_diff', ['max', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'lon_diff', ['max', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    t = group_feature(df, 'ID', 'v_diff', ['max', 'mean', 'std', 'median'])
    train = pd.merge(train, t, on='ID', how='left')

    train['lat_max_min'] = train['lat_max'] - train['lat_min']
    train['lon_max_min'] = train['lon_max'] - train['lon_min']
    train['lon_max_lat_min'] = train['lon_max'] - train['lat_min']
    train['lat_max_lon_min'] = train['lat_max'] - train['lon_min']

    train['speed_mean_iqr'] = train['speed_mean'] / train['speed_iqr']

    t = group_feature(df, 'ID', 'start_lat', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')
    t = group_feature(df, 'ID', 'start_lon', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')
    t = group_feature(df, 'ID', 'end_lat', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')
    t = group_feature(df, 'ID', 'end_lon', ['mean'])
    train = pd.merge(train, t, on='ID', how='left')

    train['end_start_lat'] = train['end_lat_mean'] - train['start_lat_mean']
    train['end_start_lon'] = train['end_lon_mean'] - train['start_lon_mean']
    # train['end_x_start_y'] = train['end_x_mean'] - train['start_y_mean']
    # train['end_y_start_x'] = train['end_y_mean'] - train['start_x_mean']

    train['end_start_L2dis'] = 6371 * np.arccos(
        np.sin(train['end_lat_mean'] * np.pi / 180) * np.sin(train['start_lat_mean'] * np.pi / 180) +
        np.cos(train['end_lat_mean'] * np.pi / 180) * np.cos(train['start_lat_mean'] * np.pi / 180) *
        np.cos((train['end_lon_mean'] - train['start_lon_mean']) * np.pi / 180))

    train['total_hour_max_min'] = train['total_hour_max'] - train['total_hour_min']
    train['hour_minute_max_min'] = train['hour_minute_max'] - train['hour_minute_min']

    mode_lat = df.groupby('ID')['lat'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_lat'] = train['ID'].map(mode_lat)

    mode_lon = df.groupby('ID')['lon'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_lon'] = train['ID'].map(mode_lon)

    mode_speed = df.groupby('ID')['speed'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_speed'] = train['ID'].map(mode_speed)

    mode_direction = df.groupby('ID')['direction'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_direction'] = train['ID'].map(mode_direction)

    mode_total_hour = df.groupby('ID')['total_hour'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_total_hour'] = train['ID'].map(mode_total_hour)

    mode_dist_diff = df.groupby('ID')['dist_diff'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_dist_diff'] = train['ID'].map(mode_dist_diff)

    mode_dist_gap_1 = df.groupby('ID')['dist_gap_1'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_dist_gap_1'] = train['ID'].map(mode_dist_gap_1)

    mode_dist_gap_2 = df.groupby('ID')['dist_gap_2'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_dist_gap_2'] = train['ID'].map(mode_dist_gap_2)

    t = group_feature(df, 'ID', 'hour', ['max', 'min'])
    train = pd.merge(train, t, on='ID', how='left')

    train['distance01'] = train['total_hour_max_min'] * train['speed_mean'] * 1.852
    train['distance02'] = 6371 * np.arccos(
        np.sin(train['lat_max'] * np.pi / 180) * np.sin(train['lat_min'] * np.pi / 180) +
        np.cos(train['lat_max'] * np.pi / 180) * np.cos(train['lat_min'] * np.pi / 180) *
        np.cos((train['lon_max'] - train['lon_min']) * np.pi / 180))
    train['dist01_02_ratio'] = train['distance01'] / train['distance02']

    train['distance_ratio_01'] = train['distance01'] / train['dist_diff_sum']
    train['distance_ratio_02'] = train['distance02'] / train['dist_diff_sum']

    # train['dist_ratio_01'] = train['end_start_L1dis'] / train['distance01']
    train['dist_ratio_021'] = train['end_start_L2dis'] / train['distance01']

    train['dist_ratio_02'] = train['end_start_L2dis'] / train['distance02']
    train['dist_ratio_03'] = train['end_start_L2dis'] / train['dist_diff_sum']

    hour_nunique = df.groupby('ID')['hour'].nunique().to_dict()
    train['hour_nunique'] = train['ID'].map(hour_nunique)

    lat_nunique = df.groupby('ID')['lat'].nunique().to_dict()
    train['lat_nunique'] = train['ID'].map(lat_nunique)

    lon_nunique = df.groupby('ID')['lon'].nunique().to_dict()
    train['lon_nunique'] = train['ID'].map(lon_nunique)

    speed_nunique = df.groupby('ID')['speed'].nunique().to_dict()
    train['speed_nunique'] = train['ID'].map(speed_nunique)

    direction_nunique = df.groupby('ID')['direction'].nunique().to_dict()
    train['direction_nunique'] = train['ID'].map(direction_nunique)

    train['slope'] = train['lon_max_min'] / np.where(train['lat_max_min'] == 0, 0.001, train['lat_max_min'])

    train['area'] = train['lat_max_min'] * train['lon_max_min']

    return train


train_label = train.drop_duplicates('ID')
test_label = test.drop_duplicates('ID')

train_label['type'].value_counts(1)

type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
type_map_rev = {v: k for k, v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)

train_label = extract_feature(train, train_label)
test_label = extract_feature(test, test_label)

features = [x for x in train_label.columns if x not in ['ID', 'type', 'total_hour_max', 'total_hour_min',
                                                        'time', 'month', 'weekday', 'day', 'hour', 'minute',
                                                        'end_start_x_mean', 'end_start_y_mean', 'total_hour',
                                                        ]]
target = 'type'

print(len(features), ','.join(features))

label = train_label[(train_label['type'] == 0)]
train_label = pd.concat([train_label, label])
train_label = pd.concat([train_label, label])

X = train_label[features].copy()
y = train_label[target]

X.fillna(0, inplace=True)
'''
llf = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='multiclass',
    n_estimators=5000,
    metrics='multi_error',
    lambda_l1=1e-4,
    lambda_l2=1e-4,
    min_child_samples=50,
    bagging_fraction=0.75,
    num_class=3,
)

lgb_pred = np.zeros((len(test_label), 3))
lgb_oof = np.zeros((len(X), 3))
sk = StratifiedKFold(n_splits=7, shuffle=True, random_state=2019)
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

    test_pred = llf.predict_proba(test_label[features])
    lgb_pred += test_pred / 7

lgb_oof_1 = np.argmax(lgb_oof, axis=1)
print('oof f1', metrics.f1_score(lgb_oof_1, y, average='macro'))
'''
xlf = xgb.XGBClassifier(
    n_estimators=5000,
    objective='multi:softprob',
    reg_alpha=1e-4,
    reg_lambda=1e-4,
    min_child_samples=50,
    bagging_fraction=0.7,
    num_class=3,
)
xgb_pred = np.zeros((len(test_label), 3))
xgb_oof = np.zeros((len(X), 3))
sk = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
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

    test_pred = xlf.predict_proba(test_label[features])
    xgb_pred += test_pred / 10

xgb_oof_1 = np.argmax(xgb_oof, axis=1)
print('oof f1', metrics.f1_score(xgb_oof_1, y, average='macro'))
oof = xgb_oof
oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))

pred = np.zeros((len(test_label), 3))
pred = xgb_pred
'''
oof = lgb_oof
oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))

pred = np.zeros((len(test_label),3))
pred = lgb_pred
'''
pred = np.argmax(pred, axis=1)
sub = test_label[['ID']]
sub['pred'] = pred

# print(sub['pred'].value_counts(1))

sub['ID'] = sub['ID']
sub['pred'] = sub['pred'].map(type_map_rev)
sub.sort_values("ID", inplace=True)
sub.to_csv("result.csv", index=None, header=None)