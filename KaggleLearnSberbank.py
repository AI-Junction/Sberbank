# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:27:51 2017

@author: echtpar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


color = sns.color_palette()

#%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
run1 = False
run2 = False
run3 = False

train_df = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv", parse_dates=['timestamp'])
id_test = list(df_test['id'])


train_df.shape

train_df.head()

print(train_df.shape[0])

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()


plt.figure(figsize=(12,8))
dfsnsplt = pd.DataFrame(train_df.price_doc.values.astype(int))
print(type(dfsnsplt))
sns.distplot(dfsnsplt, bins=60, kde=True)
#sns.distplot(train_df.price_doc.values)
plt.xlabel('price', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()


train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: str(x)[:4]+str(x)[5:7])
train_df['yearmonth'] = train_df['yearmonth'].astype(int)

df_test['yearmonth'] = df_test['timestamp'].apply(lambda x: str(x)[:4]+str(x)[5:7])
df_test['yearmonth'] = df_test['yearmonth'].astype(int)


grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

#train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

missing_df_temp = train_df.isnull()
missing_df_temp.head()
missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values in training data")
ax.set_title("Number of missing values in each column")
plt.show()


missing_df_temp = df_test.isnull()
missing_df_temp.head()
missing_df = df_test.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values in test data")
ax.set_title("Number of missing values in each column")
plt.show()



for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))


for f in df_test.columns:
    if df_test[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_test[f].values)) 
        df_test[f] = lbl.transform(list(df_test[f].values))
        


#clean data
bad_index = train_df[train_df.life_sq > train_df.full_sq].index
train_df.ix[bad_index, "life_sq"] = np.NaN
bad_index = df_test[df_test.life_sq > df_test.full_sq].index
df_test.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
df_test.ix[equal_index, "life_sq"] = df_test.ix[equal_index, "full_sq"]

bad_index = train_df[train_df.life_sq < 5].index
train_df.ix[bad_index, "life_sq"] = np.NaN
bad_index = df_test[df_test.life_sq < 5].index
df_test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train_df[train_df.full_sq < 5].index
train_df.ix[bad_index, "full_sq"] = np.NaN
bad_index = df_test[df_test.full_sq < 5].index
df_test.ix[bad_index, "full_sq"] = np.NaN

kitch_is_build_year = [13117]
train_df.ix[kitch_is_build_year, "build_year"] = train_df.ix[kitch_is_build_year, "kitch_sq"]

bad_index = train_df[train_df.kitch_sq >= train_df.life_sq].index
train_df.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = df_test[df_test.kitch_sq >= df_test.life_sq].index
df_test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train_df[(train_df.kitch_sq == 0).values + (train_df.kitch_sq == 1).values].index
train_df.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = df_test[(df_test.kitch_sq == 0).values + (df_test.kitch_sq == 1).values].index
df_test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train_df[(train_df.full_sq > 210) & (train_df.life_sq / train_df.full_sq < 0.3)].index
train_df.ix[bad_index, "full_sq"] = np.NaN
bad_index = df_test[(df_test.full_sq > 150) & (df_test.life_sq / df_test.full_sq < 0.3)].index
df_test.ix[bad_index, "full_sq"] = np.NaN

bad_index = train_df[train_df.life_sq > 300].index
train_df.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = df_test[df_test.life_sq > 200].index
df_test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

train_df.product_type.value_counts(normalize= True)
df_test.product_type.value_counts(normalize= True)

bad_index = train_df[train_df.build_year < 1500].index
train_df.ix[bad_index, "build_year"] = np.NaN
bad_index = df_test[df_test.build_year < 1500].index
df_test.ix[bad_index, "build_year"] = np.NaN

bad_index = train_df[train_df.num_room == 0].index 
train_df.ix[bad_index, "num_room"] = np.NaN
bad_index = df_test[df_test.num_room == 0].index 
df_test.ix[bad_index, "num_room"] = np.NaN

bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train_df.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
df_test.ix[bad_index, "num_room"] = np.NaN

bad_index = train_df[(train_df.floor == 0).values * (train_df.max_floor == 0).values].index
train_df.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train_df[train_df.floor == 0].index
train_df.ix[bad_index, "floor"] = np.NaN

bad_index = train_df[train_df.max_floor == 0].index
train_df.ix[bad_index, "max_floor"] = np.NaN
bad_index = df_test[df_test.max_floor == 0].index
df_test.ix[bad_index, "max_floor"] = np.NaN

bad_index = train_df[train_df.floor > train_df.max_floor].index
train_df.ix[bad_index, "max_floor"] = np.NaN
bad_index = df_test[df_test.floor > df_test.max_floor].index
df_test.ix[bad_index, "max_floor"] = np.NaN

"""
train_df.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train_df.ix[bad_index, "floor"] = np.NaN

train_df.material.value_counts()
df_test.material.value_counts()
train_df.state.value_counts()

bad_index = train_df[train_df.state == 33].index
train_df.ix[bad_index, "state"] = np.NaN
df_test.state.value_counts()
"""

ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit



col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit


train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)
train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['full_sq'].astype(float)

df_test['rel_floor'] = df_test['floor'] / df_test['max_floor'].astype(float)
df_test['rel_kitch_sq'] = df_test['kitch_sq'] / df_test['full_sq'].astype(float)

#train_df.apartment_name=train_df.sub_area + train_df['metro_km_avto'].astype(str)
#df_test.apartment_name=df_test.sub_area + df_test['metro_km_avto'].astype(str)

train_df['room_size'] = train_df['life_sq'] / train_df['num_room'].astype(float)
df_test['room_size'] = df_test['life_sq'] / df_test['num_room'].astype(float)


# Add month-year

month_year = (train_df.timestamp.dt.month.astype(int) + (train_df.timestamp.dt.year * 100).astype(int))
month_year_cnt_map = month_year.value_counts().to_dict()
train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (df_test.timestamp.dt.month.astype(int) + (df_test.timestamp.dt.year * 100).astype(int))
month_year_cnt_map = month_year.value_counts().to_dict()
df_test['month_year_cnt'] = month_year.map(month_year_cnt_map)


# Add week-year count
week_year = (train_df.timestamp.dt.weekofyear.astype(int) + (train_df.timestamp.dt.year * 100).astype(int))
week_year_cnt_map = week_year.value_counts().to_dict()
train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (df_test.timestamp.dt.weekofyear.astype(int) + (df_test.timestamp.dt.year * 100).astype(int))
week_year_cnt_map = week_year.value_counts().to_dict()
df_test['week_year_cnt'] = week_year.map(week_year_cnt_map)


# Add month and day-of-week
train_df['month'] = train_df.timestamp.dt.month
train_df['dow'] = train_df.timestamp.dt.dayofweek

df_test['month'] = df_test.timestamp.dt.month
df_test['dow'] = df_test.timestamp.dt.dayofweek


# Other feature engineering
train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)
train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['full_sq'].astype(float)

df_test['rel_floor'] = df_test['floor'] / df_test['max_floor'].astype(float)
df_test['rel_kitch_sq'] = df_test['kitch_sq'] / df_test['full_sq'].astype(float)

#train_df.apartment_name=train_df.sub_area + train_df['metro_km_avto'].astype(str)
#df_test.apartment_name=df_test.sub_area + train_df['metro_km_avto'].astype(str)

train_df['room_size'] = train_df['life_sq'] / train_df['num_room'].astype(float)
df_test['room_size'] = df_test['life_sq'] / df_test['num_room'].astype(float)


train_df.drop(['timestamp', 'id'], axis = 1, inplace = True)
df_test.drop(['timestamp', 'id'], axis = 1, inplace = True)

train_y = train_df.price_doc.values
train_df.drop(['price_doc'], axis = 1, inplace = True)
train_X = train_df

train_df.columns.values

print(train_X.shape)
print(train_y.shape)

print(df_test.shape)

train_X1 = np.nan_to_num(train_X[:25000]).astype(int)
val_X = np.nan_to_num(train_X[25000:]).astype(int)
train_y1 = np.nan_to_num(train_y[:25000]).astype(int)
val_y = np.nan_to_num(train_y[25000:]).astype(int)

train_X = np.nan_to_num(train_X).astype(int)
train_y = np.nan_to_num(train_y).astype(int)

ylog_train = np.log1p(train_y)


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(train_X, train_y)
dtest = xgb.DMatrix(np.array(df_test))


# using XGBoost
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=500, early_stopping_rounds=20, verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

#num_boost_rounds = len(cv_output)
num_boost_rounds = 350
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

#print(len(cv_output))

#y_log_predict = model.predict(dtest)
y_predict = model.predict(dtest)
#y_predict = np.exp(y_log_predict) - 1
y_predict = np.round(y_predict * 1.008)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

# Using linear regression - results were not good
#from sklearn.model_selection import cross_val_predict
#from sklearn import linear_model


# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
#lr = linear_model.LinearRegression()
#predicted = cross_val_predict(lr, train_X, train_y, cv=10)
#lr.fit(train_X, train_y)
#y_predict = lr.predict(np.nan_to_num(df_test))

#y_predict = model.predict(dtest)
#y_predict = np.round(y_predict * 0.99)
#print(type(y_predict))
#print(len(y_predict))
#print(len(id_test))

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
df_sub.head()
result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
print(result.head())
result["price_doc"] = np.exp( .75*np.log(result.price_doc_louis) + .25*np.log(result.price_doc_bruno))
result.drop(["price_doc_louis","price_doc_bruno"],axis=1,inplace=True)
result.head()
result.to_csv('CP_naive_xgb_V10.csv', index=False)



#output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
#output.head()
#output.to_csv('CP_naive_xgb_without_fe_V8.csv', index=False)


