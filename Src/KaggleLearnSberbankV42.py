# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# See notebook
#   https://www.kaggle.com/aharless/probabilistic-version-of-small-improvements/notebook
# for some explanation of small improvements

# Parameters
prediction_stderr = 0.0073 #  assumed standard error of predictions
                          #  (smaller values make output closer to input)
train_test_logmean_diff = 0.1 # assumed shift used to adjust frequencies for time trend
probthresh = 90  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero



# RUN THE MODELS


#########################

# winning Kernel by SchoolPal given below

# change in main branch

#########################

# change in helper branch
# 2nd change in helper branch
from sklearn.model_selection import train_test_split,KFold,TimeSeriesSplit
from sklearn import model_selection, preprocessing
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import model_selection, preprocessing
import pdb

'''
macro = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv", parse_dates=['timestamp'])
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv", parse_dates=['timestamp'])
'''

macro = pd.read_csv("src/data/raw/sberbank_macro.csv")
train = pd.read_csv("src/data/raw/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("src/data/raw/test.csv", parse_dates=['timestamp'])



def process(train,test):
    RS=1
    np.random.seed(RS)
    ROUNDS = 1500 # 1300,1400 all works fine
    params = {
        'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.01 , #small learn rate, large number of iterations
            'verbose': 0,
            'num_leaves': 2 ** 5,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': RS,
            'feature_fraction': 0.7,
            'feature_fraction_seed': RS,
            'max_bin': 100,
            'max_depth': 7,
            'num_rounds': ROUNDS,
        }
    #Remove the bad prices as suggested by Radar
    train=train[(train.price_doc>1e6) & (train.price_doc!=2e6) & (train.price_doc!=3e6)]
    train.loc[(train.product_type=='Investment') & (train.build_year<2000),'price_doc']*=0.9 
    train.loc[train.product_type!='Investment','price_doc']*=0.969 #Louis/Andy's magic number
#    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

  
    id_test = test.id
    times=pd.concat([train.timestamp,test.timestamp])
    num_train=train.shape[0]
    y_train = train["price_doc"]
    train.drop(['price_doc'],inplace=True,axis=1)
    da=pd.concat([train,test])
    da['na_count']=da.isnull().sum(axis=1)
    df_cat=None
    to_remove=[]
    for c in da.columns:
        if da[c].dtype=='object':
            oh=pd.get_dummies(da[c],prefix=c)
            if df_cat is None:
                df_cat=oh
            else:
                df_cat=pd.concat([df_cat,oh],axis=1)
            to_remove.append(c)
    da.drop(to_remove,inplace=True,axis=1)

    #Remove rare features,prevent overfitting
    to_remove=[]
    if df_cat is not None:
        sums=df_cat.sum(axis=0)
        to_remove=sums[sums<200].index.values
        df_cat=df_cat.loc[:,df_cat.columns.difference(to_remove)]
        da = pd.concat([da, df_cat], axis=1)
    x_train=da[:num_train].drop(['timestamp','id'],axis=1)
    x_test=da[num_train:].drop(['timestamp','id'],axis=1)
    #Log transformation, boxcox works better.
    y_train=np.log(y_train)
    train_lgb=lgb.Dataset(x_train,y_train)
    model=lgb.train(params,train_lgb,num_boost_round=ROUNDS)
    predict=model.predict(x_test)
    predict=np.exp(predict)
    return predict,id_test

    
#    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])
#    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])
predict,id_test=process(train,test)
output=pd.DataFrame({'id':id_test,'price_doc':predict})
output.to_csv('lgb.csv',index=False)


#########################

# older Kernels given below

#########################


import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from scipy.stats import norm


#load files
'''
macro = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv", parse_dates=['timestamp'])
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv", parse_dates=['timestamp'])
'''


macro = pd.read_csv("src/data/raw/sberbank_macro.csv")
train = pd.read_csv("src/data/raw/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("src/data/raw/test.csv", parse_dates=['timestamp'])

            
#train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
#test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
id_test = test.id

#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index
test.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.loc[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN
test.state.value_counts()

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
#print(month_year[:10])
#print(train.timestamp.dt.month[:10])
#print(train.timestamp.dt.year[:10])
month_year_cnt_map = month_year.value_counts().to_dict()
#print(month_year_cnt_map.keys())
#print(month_year_cnt_map.values())

train['month_year_cnt'] = month_year.map(month_year_cnt_map)
#print(train['month_year_cnt'][:10])


month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

#print(train['timestamp'].dtype)

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = .05 + train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = .05 + train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = .05 + test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = .05 + test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]
test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]


#########################################################################

# Aggreagte house price data derived from 
# http://www.globalpropertyguide.com/real-estate-house-prices/R#russia
# by luckyzhou
# See https://www.kaggle.com/luckyzhou/lzhou-test/comments

rate_2015_q2 = 1
rate_2015_q1 = rate_2015_q2 / 0.9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104  # This is 1.002 (relative to mult), close to 1:
rate_2012_q4 = rate_2013_q1 / 0.9832  #     maybe use 2013q1 as a base quarter and get rid of mult?
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011


# train 2015
train['average_q_price'] = 1

train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2014
train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


# train 2013
train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


# train 2012
train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


# train 2011
train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

train['price_doc'] = train['price_doc'] * train['average_q_price']


#########################################################################################################

mult = 1.054 # Trying another magic number
train['price_doc'] = train['price_doc'] * mult
y_train = train["price_doc"]
print(len(y_train))

#########################################################################################################


def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])
   train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])
   train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

num_train = len(train)
print(num_train)

df_all = pd.concat([train, test])

df_all = df_all.join(macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

#for c in x_all.columns:
#    if x_all[c].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(x_all[c].values))
#        x_all[c] = lbl.transform(list(x_all[c].values))

#x_train = x_all[:num_train]
#x_test = x_all[num_train:]


factorize = lambda t: pd.factorize(t[1])[0]

df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

print(len(df_values))


df_values = df_values.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
#test = test.drop(["id", "timestamp"], axis=1)

# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

x_train = X_all[:num_train]
x_test = X_all[num_train:]

print(type(x_train))

#for c in x_train.columns:
#    if x_train[c].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(x_train[c].values))
#        x_train[c] = lbl.transform(list(x_train[c].values))
#
#for c in x_test.columns:
#    if x_test[c].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(x_test[c].values))
#        x_test[c] = lbl.transform(list(x_test[c].values))






#xgb_params = {
#    'eta': 0.05,
#    'max_depth': 6,
#    'subsample': 0.6,
#    'colsample_bytree': 1,
#    'objective': 'reg:linear',
#    'eval_metric': 'rmse',
#    'silent': 1
#}


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}



dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


y_predict = model.predict(dtest)
gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

gunja_output.to_csv('2017-06-26-revisedGunjaV2.csv', index=False)


print(x_train.shape)
print(y_train.shape)


######################################################################################################


#macro = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv")


#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')
id_test = test.id

mult = .969

y_train = train["price_doc"] * mult + 10
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}




dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 385  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


#######################################################################################################

df_macro = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
df_train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv", parse_dates=['timestamp'])


#df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
#df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
#df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = 0.969
y_train = df_train['price_doc'].values * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
print(num_train)
df_all = pd.concat([df_train, df_test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]
test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])
   train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])
   train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])
   test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])
   test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

print(len(X_train))

c = []
a = [1,2,3,4,5]
b = [6,7,8,9]
c = a + b
print(c)
print(len(a))
print(c[:5])
print(c[5:])




# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

print(len(df_values))

# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

#X_train_forCP = X_all[:num_train]
#X_test_forCP = X_all[num_train:]
#y_train_forCP = y_train

#print(X_train_forCP.shape)
#print(X_test_forCP.shape)
#print(y_train_forCP.shape)

df_columns = df_values.columns


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

num_boost_rounds = 420  # From Bruno's original CV, I think
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})


####################################################################################################3



first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
                                    .286*np.log(first_result.price_doc_bruno) ) 
result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) +
                              .22*np.log(result.price_doc_gunja) )
                              
result["price_doc"] =result["price_doc"] *0.9915        
result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
result.head()
result.to_csv('same_result.csv', index=False)





# APPLY PROBABILISTIC IMPROVEMENTS

#df_macro = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv")



preds = result
#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')



# Select investment sales from training set and generate frequency distribution
invest = train[train.product_type=="Investment"]
freqs = invest.price_doc.value_counts().sort_index()

# Select investment sales from test set predictions
test_invest_ids = test[test.product_type=="Investment"]["id"]
invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")
print(invest_preds.shape)


# Express X-axis of training set frequency distribution as logarithms, 
#    and save standard deviation to help adjust frequencies for time trend.
lnp = np.log(invest.price_doc)
stderr = lnp.std()
lfreqs = lnp.value_counts().sort_index()
print(lfreqs.shape)



# Adjust frequencies for time trend
lnp_diff = train_test_logmean_diff
lnp_mean = lnp.mean()
lnp_newmean = lnp_mean + lnp_diff

def norm_diff(value):
    return norm.pdf((value-lnp_diff)/stderr) / norm.pdf(value/stderr)

newfreqs = lfreqs * (pd.Series(lfreqs.index.values-lnp_newmean).apply(norm_diff).values)

# Logs of model-predicted prices
lnpred = np.log(invest_preds.price_doc)

# Create assumed probability distributions
stderr = prediction_stderr
mat =(np.array(newfreqs.index.values)[:,np.newaxis] - np.array(lnpred)[np.newaxis,:])/stderr
modelprobs = norm.pdf(mat)

# Multiply by frequency distribution.
freqprobs = pd.DataFrame( np.multiply( np.transpose(modelprobs), newfreqs.values ) )
freqprobs.index = invest_preds.price_doc.values
freqprobs.columns = freqs.index.values.tolist()

# Find mode for each case.
prices = freqprobs.idxmax(axis=1)

# Apply threshold to exclude low-confidence cases from recoding
priceprobs = freqprobs.max(axis=1)
mask = priceprobs<probthresh
prices[mask] = np.round(prices[mask].index,-rounder)

# Data frame with new predicitons
newpricedf = pd.DataFrame( {"id":test_invest_ids.values, "price_doc":prices} )

# Merge these new predictions (for just investment properties) back into the full prediction set.
newpreds = preds.merge(newpricedf, on="id", how="left", suffixes=("_old",""))
newpreds.loc[newpreds.price_doc.isnull(),"price_doc"] = newpreds.price_doc_old
newpreds.drop("price_doc_old",axis=1,inplace=True)
newpreds.head()

newpreds.to_csv('different_result.csv', index=False)


#################################
#                               #
# CP - using Deep Learning MLP  #
#                               #
#################################

#df_macro_cp = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
#df_train_cp = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv", parse_dates=['timestamp'])
#df_test_cp = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv", parse_dates=['timestamp'])
#
#y_train = pd.DataFrame(y_train_forCP)
#x_train = X_train_forCP
#features_test = X_test_forCP
#
#print(X_train_forCP[:10, (1)])
#
#
#print(len(y_train))
#print(x_train.shape)
#print(features_test.shape)
##print(df_train_cp.shape)
#
#X_train_df = pd.DataFrame(x_train)
#
#print(X_train_forCP[:10])
#
##for col in X_train_df:
##    X_train_df[col] = X_train_df[col]/max(X_train_df[col])
#
#X_test_df = pd.DataFrame(features_test)

#for col in X_test_df:
#    X_test_df[col] = X_test_df[col]/max(X_test_df[col])
    
#features_test = np.array(X_test_df)

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score 
from sklearn.metrics import explained_variance_score
import matplotlib.pylab as plt

#X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, 
#                                                    test_size = 0.2, random_state = 1)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size = 0.2, random_state = 1)

input_dim = X_train.shape[1]
print(input_dim)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

model = Sequential()

model.add(Dense(128, input_dim=input_dim, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu')) 

model.add(Dropout(0.1))

model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu')) 

model.add(Dense(1, activation='relu' ))

    
#    model.add(Dense(128, input_dim=32, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
#    
#    model.add(Dropout(0.1))
#    
#    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
#    
#    model.add(Dropout(0.1))
#    
#    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(32, kernel_initializer='normal', activation='relu')) 
    
#    model.add(Dropout(0.1))
    
#    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(16, kernel_initializer='normal', activation='relu')) 
    
#    model.add(Dense(1, kernel_initializer='normal'))
    
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
#    return model


#mlp_model = mlp_model()

print(X_train.shape[0])

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

print(model.layers)

mlp_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=10, batch_size=16, verbose=1)

#loss_plot(mlp_history)
#mae_plot(mlp_history)

X_train.shape
X_test.shape
y_train_mlp = model.predict(np.array(X_train))
y_test_mlp = model.predict(np.array(X_val))

print(y_train_mlp.shape)
print(y_test_mlp.shape)

print(y_train_mlp[:10])
print(y_test_mlp[:10])


def loss_plot(fit_history):
    plt.figure(figsize=(14, 6))

    plt.plot(fit_history.history['loss'], label = 'train')
    plt.plot(fit_history.history['val_loss'], label = 'test')

    plt.legend()
    plt.title('Loss Function');  
    
def mae_plot(fit_history):
    plt.figure(figsize=(14, 6))

    plt.plot(fit_history.history['mean_absolute_error'], label = 'train')
    plt.plot(fit_history.history['val_mean_absolute_error'], label = 'test')

    plt.legend()
    plt.title('Mean Absolute Error'); 

def scores(regressor, y_train, y_test, y_train_reg, y_test_reg):
    print("_______________________________________")
    print(regressor)
    print("_______________________________________")
    print("EV score. Train: ", explained_variance_score(y_train, y_train_reg))
    print("EV score. Test: ", explained_variance_score(y_test, y_test_reg))
    print("---------")
    print("R2 score. Train: ", r2_score(y_train, y_train_reg))
    print("R2 score. Test: ", r2_score(y_test, y_test_reg))
    print("---------")
    print("MSE score. Train: ", mean_squared_error(y_train, y_train_reg))
    print("MSE score. Test: ", mean_squared_error(y_test, y_test_reg))
    print("---------")
    print("MAE score. Train: ", mean_absolute_error(y_train, y_train_reg))
    print("MAE score. Test: ", mean_absolute_error(y_test, y_test_reg))
    print("---------")
    print("MdAE score. Train: ", median_absolute_error(y_train, y_train_reg))
    print("MdAE score. Test: ", median_absolute_error(y_test, y_test_reg))


loss_plot(mlp_history)
mae_plot(mlp_history)
scores('MLP Model', y_train, y_val, y_train_mlp, y_test_mlp)


X_test.shape
target_test_predict = model.predict(X_test)

print(X_train[:10])

print(target_test_predict[:10])

submission = pd.DataFrame({"id": test['id'], "price_doc": target_test_predict})
print(submission[0:5])

submission.to_csv('2017-06-26-keras_sberbank_housing.csv', index=False)
