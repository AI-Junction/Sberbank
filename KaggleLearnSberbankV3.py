# -*- coding: utf-8 -*-

"""
Chandrakant Pattekar
Notebook Sberbank - Russian Housing Market

forked from Sberbank Russian Housing Market by Olga Belitskaya (+0/-0/~0)
Chandrakant Pattekar
Sberbank Russian Housing Market
last run a few seconds ago · Python notebook
using data from Sberbank Russian Housing Market ·
PrivateMake Public

"""

import numpy as np 
import pandas as pd 
import scipy

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns
import matplotlib.pylab as plt

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import KFold, ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

import keras as ks
from keras.models import Sequential, load_model, Model
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasRegressor

macro = pd.read_csv('../input/macro.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

macro[100:103].T[1:15]

train[200:203].T[1:15]

X_list_num = ['full_sq', 'num_room', 'floor', 
              'preschool_education_centers_raion', 'school_education_centers_raion', 
              'children_preschool', 'children_school',
              'shopping_centers_raion', 'healthcare_centers_raion', 
              'office_raion', 'sport_objects_raion',
              'metro_min_walk', 'public_transport_station_min_walk', 
              'railroad_station_walk_min', 'cafe_count_500',
              'kremlin_km', 'workplaces_km', 'ID_metro',
              'public_healthcare_km', 'kindergarten_km', 'school_km', 'university_km', 
              'museum_km', 'fitness_km', 'park_km', 'shopping_centers_km',
              'additional_education_km', 'theater_km', 
              'raion_popul', 'work_all', 'young_all', 'ekder_all']
X_list_cat = ['sub_area', 'ecology', 'ID_metro', 'big_market_raion']

features_train = train[X_list_num]
features_test = test[X_list_num]
target_train = train['price_doc']

plt.style.use('seaborn-bright')
plt.figure(figsize=(14, 6))

plt.hist(np.log(target_train), bins=200, normed=True, alpha=0.7)

plt.xlabel("Prices")
plt.title('Sberbank Russian Housing Data');

print ("Sberbank Russian Housing Dataset Statistics: \n")
print ("Number of houses = ", len(target_train))
print ("Number of features = ", len(list(features_train.keys())))
print ("Minimum house price = ", np.min(target_train))
print ("Maximum house price = ", np.max(target_train))
print ("Mean house price = ", "%.2f" % np.mean(target_train))
print ("Median house price = ", "%.2f" % np.median(target_train))
print ("Standard deviation of house prices =", "%.2f" % np.std(target_train))

features_train.isnull().sum()

features_test.isnull().sum()

df = pd.DataFrame(features_train, columns=X_list_num)
df['prices'] = target_train

df = df.dropna(subset=['num_room'])

df['metro_min_walk'] = df['metro_min_walk'].interpolate(method='quadratic')
features_test['metro_min_walk'] = features_test['metro_min_walk'].interpolate(method='quadratic')

df['railroad_station_walk_min'] = df['railroad_station_walk_min'].interpolate(method='quadratic')
features_test['railroad_station_walk_min'] = \
features_test['railroad_station_walk_min'].interpolate(method='quadratic')

df['floor'] = df['floor'].fillna(df['floor'].median())
len(df)

ID_metro_cat = pd.factorize(df['ID_metro'])
df['ID_metro'] = ID_metro_cat[0]

ID_metro_pairs = dict(zip(list(ID_metro_cat[1]), list(set(ID_metro_cat[0]))))
ID_metro_pairs[224] = 219
features_test['ID_metro'].replace(ID_metro_pairs,inplace=True)

pearson = df.corr(method='pearson')
corr_with_prices = pearson.ix[-1][:-1]
corr_with_prices[abs(corr_with_prices).argsort()[::-1]]

target_train = df['prices'].as_matrix()
features_train = df.drop('prices', 1).as_matrix()



X_train, X_test, y_train, y_test = train_test_split(features_train, target_train, 
                                                    test_size = 0.2, random_state = 1)
X_train.shape, X_test.shape

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

def regression(regressor, x_train, x_test, y_train):
    reg = regressor
    reg.fit(x_train, y_train)
    
    y_train_reg = reg.predict(x_train)
    y_test_reg = reg.predict(x_test)
    
    return y_train_reg, y_test_reg
    
y_train_gbr, y_test_gbr = regression(GradientBoostingRegressor(n_estimators=100, 
                                                               max_depth=3, 
                                                               learning_rate=.1, 
                                                               min_samples_leaf=5, 
                                                               min_samples_split=5), 
                                     X_train, X_test, y_train)

y_train_br, y_test_br = regression(BaggingRegressor(n_estimators=100), 
                                   X_train, X_test, y_train)

plt.figure(figsize = (12,6))

plt.plot(y_test[1:50], color = 'black', label='Real Data')

plt.plot(y_test_gbr[1:50], label='Gradient Boosting')
plt.plot(y_test_br[1:50], label='Bagging')

plt.legend()
plt.title("Regressors' Predictions vs Real Data"); 


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

def mlp_model():
    model = Sequential()
    
    model.add(Dense(128, input_dim=32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    
    model.add(Dropout(0.1))
    
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    
    model.add(Dropout(0.1))
    
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu')) 
    
#    model.add(Dropout(0.1))
    
#    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(16, kernel_initializer='normal', activation='relu')) 
    
    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    return model


mlp_model = mlp_model()

mlp_history = mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            nb_epoch=80, batch_size=16, verbose=0)

loss_plot(mlp_history)
mae_plot(mlp_history)

y_train_mlp = mlp_model.predict(X_train)
y_test_mlp = mlp_model.predict(X_test)

plt.figure(figsize = (12,6))

plt.plot(y_test[1:50], color = 'black', label='Real Data')

plt.plot(y_test_mlp[1:50], label='MLP')
# plt.plot(y_test_br[1:50], label='CNN')

plt.legend()
plt.title("Neural Networks' Predictions vs Real Data"); 

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


scores('Gradient Boosting Regressor', y_train, y_test, y_train_gbr, y_test_gbr)
scores('Bagging Regressor', y_train, y_test, y_train_br, y_test_br)

scores('MLP Model', y_train, y_test, y_train_mlp, y_test_mlp)

scale = StandardScaler()
features_train = scale.fit_transform(features_train)
features_test = scale.fit_transform(features_test.as_matrix())


reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=.1, 
                                min_samples_leaf=5, min_samples_split=5)
reg.fit(features_train, target_train)

target_train_predict = reg.predict(features_train)
target_test_predict = reg.predict(features_test)

print("_______________________________________")
print("Gradient Boosting  Regressor")
print("_______________________________________")
print("EV score. Train: ", explained_variance_score(target_train, target_train_predict))
print("---------")
print("R2 score. Train: ", r2_score(target_train, target_train_predict))
print("---------")
print("MSE score. Train: ", mean_squared_error(target_train, target_train_predict))
print("---------")
print("MAE score. Train: ", mean_absolute_error(target_train, target_train_predict))
print("---------")
print("MdAE score. Train: ", median_absolute_error(target_train, target_train_predict))

pd.set_option('display.float_format', lambda x: '%.2f' % x)
target_predict = ["{0:.2f}".format(x) for x in target_test_predict]

submission = pd.DataFrame({"id": test['id'], "price_doc": target_predict})
print(submission[0:5])

submission.to_csv('kaggle_sberbank_housing.csv', index=False)






    























"""
Chandrakant Pattekar
latest iteration in this silly game

forked from latest iteration in this silly game by Jayaraman (+0/-0/~0)
Chandrakant Pattekar
Sberbank Russian Housing Market
last run an hour ago · Python script
using data from Sberbank Russian Housing Market ·
PrivateMake Public
"""


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

#load files
#train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
#test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
#id_test = test.id


train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\train.csv", parse_dates=['timestamp'])
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\test.csv", parse_dates=['timestamp'])
#macro = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllSberbankData\\macro.csv")
id_test = test.id

print(train.index.dtype)

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
train.loc[train.full_sq == 0, 'full_sq'] = 52
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

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

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

rate_2016_q2 = 1
rate_2016_q1 = rate_2016_q2 / .9990
rate_2015_q4 = rate_2016_q1 / .983
rate_2015_q3 = rate_2015_q4 / .9834
rate_2015_q2 = rate_2015_q3 / .9815
rate_2015_q1 = rate_2015_q2 / .9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104
rate_2012_q4 = rate_2013_q1 / 0.9832
rate_2012_q3 = rate_2012_q4 / 1.0279
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011

# test data
test['average_q_price'] = 1

test_2016_q2_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month <= 7].index
test.loc[test_2016_q2_index, 'average_q_price'] = rate_2016_q2
# test.loc[test_2016_q2_index, 'year_q'] = '2016_q2'

test_2016_q1_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 1].loc[test['timestamp'].dt.month < 4].index
test.loc[test_2016_q1_index, 'average_q_price'] = rate_2016_q1
# test.loc[test_2016_q2_index, 'year_q'] = '2016_q1'

test_2015_q4_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 10].loc[test['timestamp'].dt.month < 12].index
test.loc[test_2015_q4_index, 'average_q_price'] = rate_2015_q4
# test.loc[test_2015_q4_index, 'year_q'] = '2015_q4'

test_2015_q3_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 7].loc[test['timestamp'].dt.month < 10].index
test.loc[test_2015_q3_index, 'average_q_price'] = rate_2015_q3
# test.loc[test_2015_q3_index, 'year_q'] = '2015_q3'

# test_2015_q2_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index
# test.loc[test_2015_q2_index, 'average_q_price'] = rate_2015_q2

# test_2015_q1_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index
# test.loc[test_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2015
train['average_q_price'] = 1

train_2015_q4_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2015_q4_index, 'price_doc'] = train.loc[train_2015_q4_index, 'price_doc'] * rate_2015_q4
train.loc[train_2015_q4_index, 'average_q_price'] = rate_2015_q4

train_2015_q3_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
#train.loc[train_2015_q3_index, 'price_doc'] = train.loc[train_2015_q3_index, 'price_doc'] * rate_2015_q3
train.loc[train_2015_q3_index, 'average_q_price'] = rate_2015_q3

train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
#train.loc[train_2015_q2_index, 'price_doc'] = train.loc[train_2015_q2_index, 'price_doc'] * rate_2015_q2
train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
#train.loc[train_2015_q1_index, 'price_doc'] = train.loc[train_2015_q1_index, 'price_doc'] * rate_2015_q1
train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2014
train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
#train.loc[train_2014_q4_index, 'price_doc'] = train.loc[train_2014_q4_index, 'price_doc'] * rate_2014_q4
train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
#train.loc[train_2014_q3_index, 'price_doc'] = train.loc[train_2014_q3_index, 'price_doc'] * rate_2014_q3
train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
#train.loc[train_2014_q2_index, 'price_doc'] = train.loc[train_2014_q2_index, 'price_doc'] * rate_2014_q2
train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
#train.loc[train_2014_q1_index, 'price_doc'] = train.loc[train_2014_q1_index, 'price_doc'] * rate_2014_q1
train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


# train 2013
train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2013_q4_index, 'price_doc'] = train.loc[train_2013_q4_index, 'price_doc'] * rate_2013_q4
train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
# train.loc[train_2013_q3_index, 'price_doc'] = train.loc[train_2013_q3_index, 'price_doc'] * rate_2013_q3
train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
# train.loc[train_2013_q2_index, 'price_doc'] = train.loc[train_2013_q2_index, 'price_doc'] * rate_2013_q2
train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
# train.loc[train_2013_q1_index, 'price_doc'] = train.loc[train_2013_q1_index, 'price_doc'] * rate_2013_q1
train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


# train 2012
train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2012_q4_index, 'price_doc'] = train.loc[train_2012_q4_index, 'price_doc'] * rate_2012_q4
train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
# train.loc[train_2012_q3_index, 'price_doc'] = train.loc[train_2012_q3_index, 'price_doc'] * rate_2012_q3
train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
# train.loc[train_2012_q2_index, 'price_doc'] = train.loc[train_2012_q2_index, 'price_doc'] * rate_2012_q2
train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
# train.loc[train_2012_q1_index, 'price_doc'] = train.loc[train_2012_q1_index, 'price_doc'] * rate_2012_q1
train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


# train 2011
train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
# train.loc[train_2011_q4_index, 'price_doc'] = train.loc[train_2011_q4_index, 'price_doc'] * rate_2011_q4
train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
# train.loc[train_2011_q3_index, 'price_doc'] = train.loc[train_2011_q3_index, 'price_doc'] * rate_2011_q3
train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
# train.loc[train_2011_q2_index, 'price_doc'] = train.loc[train_2011_q2_index, 'price_doc'] * rate_2011_q2
train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
# train.loc[train_2011_q1_index, 'price_doc'] = train.loc[train_2011_q1_index, 'price_doc'] * rate_2011_q1
train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

train['price_doc'] = train['price_doc'] * train['average_q_price']
# train.drop('average_q_price', axis=1, inplace=True)

print('price changed done')

y_train = train["price_doc"]
# x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
# x_test = test.drop(["id", "timestamp"], axis=1)

x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))
        #x_train.drop(c,axis=1,inplace=True)

x_train = x_all[:num_train]
x_test = x_all[num_train:]


xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#     verbose_eval=20, show_stdv=False)
#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
# print('best num_boost_rounds = ', len(cv_output))
# num_boost_rounds = len(cv_output) 

num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)
# y_predict = np.round(y_predict)
gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# gunja_output['price_doc'] = gunja_output['price_doc'] * gunja_output['average_q_price']
# gunja_output.drop('average_q_price', axis=1, inplace=True)
# gunja_output.head()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
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

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
# print('best num_boost_rounds = ', len(cv_output))
# num_boost_rounds = len(cv_output) # 382

num_boost_rounds = 385  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# output.drop('average_q_price', axis=1, inplace=True)
# output.head()

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = 0.969
y_train = df_train['price_doc'].values * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
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


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

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

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
# print('best num_boost_rounds = ', len(cv_output))
# num_boost_rounds = len(cv_output) #

num_boost_rounds = 435 # From Bruno's original CV, I think
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.head()
first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
first_result["price_doc"] = np.exp( .715*np.log(first_result.price_doc_louis) +
                                    .285*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2
result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .7801*np.log(result.price_doc_follow) +
                              .2199*np.log(result.price_doc_gunja) )
                                     
result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
result.head()
result.to_csv('sub-silly-fixed-price-changed-local.csv', index=False)


# Mostly a lot of silliness at this point:
#   Main contribution (50%) is based on Reynaldo's script with a linear transformation of y_train
#      that happens to fit the public test data well
#      and may also fit the private test data well
#      if it reflects a macro effect
#      but almost certainly won't generalize to later data
#   Second contribution (20%) is based on Bruno do Amaral's very early entry but
#      with an outlier that I deleted early in the competition
#   Third contribution (30%) is based on a legitimate data cleaning,
#      probably by gunja agarwal (or actually by Jason Benner, it seems,
#      but there's also a small transformation applied ot the predictions,
#      so also probably not generalizable),
#   This combo being run by Andy Harless on June 4


"""

Magic number improvement


Chandrakant Pattekar
Jiwon Small Improvements for Magic Number Results

forked from Jiwon Small Improvements for Magic Number Results by Andy Harless (+0/-0/~0)
Chandrakant Pattekar
Sberbank Russian Housing Market
last run an hour ago · Python script
using data from Sberbank Russian Housing Market ·
PrivateMake Public

"""

# See notebook
#   https://www.kaggle.com/aharless/probabilistic-version-of-small-improvements/notebook
# for some explanation of small improvements

# Parameters
prediction_stderr = 0.02  #  assumed standard error of predictions
                          #  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 100  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero



# RUN THE MODELS

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from scipy.stats import norm

#load files
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
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
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

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

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

#########################################################################

# Aggreagte house price data derived from 
# http://www.globalpropertyguide.com/real-estate-house-prices/R#russia
# by luckyzhou
# See https://www.kaggle.com/luckyzhou/lzhou-test/comments

rate_2015_q2 = 1
rate_2015_q1 = rate_2015_q2 / 0.99
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

mult = 1.054880504
train['price_doc'] = train['price_doc'] * mult
y_train = train["price_doc"]

#########################################################################################################

x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
#x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

x_train = x_all[:num_train]
x_test = x_all[num_train:]


xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
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


######################################################################################################

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
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


df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = 0.969
y_train = df_train['price_doc'].values * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
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


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

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

preds = result
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Select investment sales from training set and generate frequency distribution
invest = train[train.product_type=="Investment"]
freqs = invest.price_doc.value_counts().sort_index()

# Select investment sales from test set predictions
test_invest_ids = test[test.product_type=="Investment"]["id"]
invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")

# Express X-axis of training set frequency distribution as logarithms, 
#    and save standard deviation to help adjust frequencies for time trend.
lnp = np.log(invest.price_doc)
stderr = lnp.std()
lfreqs = lnp.value_counts().sort_index()

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