import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pretty_errors
from time import process_time
import datetime
import tensorflow as tf

from sklearn.neural_network import MLPRegressor
# multi-layer Perceptron regressor
from sklearn.datasets import make_regression
# random regression
from sklearn.ensemble import GradientBoostingRegressor
# gradient Boosting for regression
from catboost import CatBoostRegressor
# gradient boosting on decision trees
from lightgbm import LGBMRegressor
# leaf-wise gradient boosting model
from xgboost import XGBRegressor
# extreme gradient boosting
from sklearn.svm import SVR
# support vector regression
from sklearn.linear_model import Lasso
# Lasso
from sklearn.ensemble import RandomForestRegressor
# Random Forest
from mlxtend.regressor import StackingCVRegressor
# StackingCV, no used

# Deal with taxi_data
def taxi_data_deal():
    # Create train/test dataset(data:without SalePrice, label:SalePrice)
    # train_lighter has 100000 data, should enough
    try:
        test_data = pd.read_csv(os.path.join(os.getcwd(), 'test.csv'))
        train_data = pd.read_csv(os.path.join(os.getcwd(), 'train_lighter.csv'))
    except FileNotFoundError:
        test_data = pd.read_csv(os.path.join(os.getcwd(), 'taxi', 'test.csv'))
        train_data = pd.read_csv(os.path.join(os.getcwd(), 'taxi', 'train_lighter.csv'))
    train_label = train_data['fare_amount']
    train_data = train_data.drop('fare_amount', axis=1)

    df = pd.concat([train_data, test_data], axis = 0)

    # train_data/test_data has been checked, no any NAN value, so skip fillna
    '''
    NANColumns = []
    i = 0
    for a in train_data.isnull().sum():
        # if this column has null values
        if a != 0:
            print(train_data.columns[i], 'loss {} values'.format(a))
            NANColumns.append(train_data.columns[i])
        i += 1
    print()
    print('train_data have {} columns'.format(i))
    print('but {} columns have null values'.format(len(NANColumns)))

    # Handmade classification
    num_list = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                'FullBath', 'HalfBath', 'Bedroom', 'Kitchen', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
    str_list = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual',
                'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
                'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'PavedDrive', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
    drop_list = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType', 'FireplaceQu']
    #Id/MasVnrType useless, Alley/PoolQC/Fence/MiscFeature/FireplaceQu too many loss

    # Replace NA value to mean/'X', and drop some columns
    print('begin replace null values:')
    print('---num gogo---')
    for num_name in num_list:
        print('{} ok'.format(num_name))
        if num_name in NANColumns:
            train_data[num_name] = train_data[num_name].replace(np.nan, np.mean(train_data[num_name]))
            test_data[num_name] = test_data[num_name].replace(np.nan, np.mean(test_data[num_name]))
    print('---str gogo---')
    for str_name in str_list:
        print('{} ok'.format(str_name))
        if str_name in NANColumns:
            train_data[str_name] = train_data[str_name].replace(np.nan, "X")
            test_data[str_name] = test_data[str_name].replace(np.nan, "X")
    print('---drop gogo---')
    for drop_name in drop_list:
        print('{} ok'.format(drop_name))
        if drop_name in NANColumns:
            train_data.drop(columns=[drop_name])
            test_data.drop(columns=[drop_name])
    '''
    # 'pickup_datetime' split and turn type
    df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
    df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')
    df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
    df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
    df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')
    df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')
    df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')

    test_key = df['key'][100000:]
    df = df.drop(['pickup_datetime'] , axis=1)
    df = df.drop(['key'] , axis=1)

    # Add some useful feather
    df['longitude_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
    df['latitude_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['distance_2D'] = (df['longitude_diff']**2 + df['latitude_diff']**2)**0.5
    df['longitude_diff_difflat'] = (df['dropoff_longitude'] - df['pickup_longitude']) * np.cos((df['dropoff_latitude'] + df['pickup_latitude']) /2 /(360/2*np.pi))
    df['latitude_diff_difflat'] = (df['dropoff_latitude'] - df['pickup_latitude']) * np.cos((df['dropoff_longitude'] + df['pickup_longitude']) /2 /(360/2*np.pi))

    print(df.info)

    # Scaler and df->train/test
    scaler = MinMaxScaler()
    df_scalered = scaler.fit_transform(df)
    train_data, test_data = df_scalered[:100000], df_scalered[100000:]

    # Split dataset and label, create complete/trainpart/crosspart data/label
    split_num = 10000
    all_num = 100000
    train_data_complete = train_data
    train_data_trainpart = train_data[0:split_num]
    train_data_crosspart = train_data[split_num:all_num]
    train_label_complete = train_label
    train_label_trainpart = train_label[0:split_num]
    train_label_crosspart = train_label[split_num:all_num]

    '''
    # Encoding, In order to distinguish numeric and categorical columns
    CATEGORICAL_COLUMNS =[]
    NUMERIC_COLUMNS =[]
    i = 0
    for a in train_data.dtypes:
        if a == float or a == int:
            NUMERIC_COLUMNS.append(train_data.columns[i])
        elif a == object:
            CATEGORICAL_COLUMNS.append(train_data.columns[i])
        i += 1

    le = LabelEncoder()
    train_data_complete[CATEGORICAL_COLUMNS]    = train_data_complete[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
    train_data_trainpart[CATEGORICAL_COLUMNS]   = train_data_trainpart[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
    train_data_crosspart[CATEGORICAL_COLUMNS]   = train_data_crosspart[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col)) #SettingWithCopyWarning
    test_data[CATEGORICAL_COLUMNS]              = test_data[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
    '''

    return train_data_complete, train_label_complete, train_data_trainpart, train_label_trainpart, train_data_crosspart, train_label_crosspart, test_data, test_key

# AI model fit, evaluate, predict store to csv
def package_ai(train_data_complete, train_label_complete, train_data_crosspart, train_label_crosspart, test_data, test_key, model, model_name):
    model.fit(train_data_complete, train_label_complete)
    print("Result of {} is {}".format(model_name, model.score(train_data_crosspart, train_label_crosspart)))

    # Output
    predicted_fare_amount_list = []
    for predicted_fare_amount in model.predict(test_data):
        predicted_fare_amount_list.append(float(predicted_fare_amount))

    output = pd.DataFrame({'key':test_key, 'fare_amount': predicted_fare_amount_list})
    output.to_csv(os.path.join(os.getcwd(), 'taxi', '{}_submissions.csv'.format(model_name)), index=False)

# tensorflow AI model create, compile, fit, evaluate, predict store to csv
def tensorflow_ai(train_data_complete, train_label_complete, train_data_crosspart, train_label_crosspart, test_data, model_type):
    model = tf.keras.Sequential()
    if model_type == 'dnn':
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
                units = 1,
                #input_shape = [1,74],
                kernel_initializer = 'ones',
                kernel_regularizer = tf.keras.regularizers.L1L2(l1=0, l2=1),
            )
        )
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.Dense(25))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(1))
    elif model_type == 'cnn':
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.Dense(25))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mae')
    train_history = model.fit(train_data_complete, train_label_complete, batch_size=8, epochs=20)

    model.summary()
    #acc = train_history.history['acc']
    #val_acc = train_history.history['val_acc']
    loss = train_history.history['loss']
    #val_loss = train_history.history['val_loss']

    epochs = range(1, len(loss)+1)
    '''
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    '''
    plt.plot(epochs, loss, 'bo', label='Training loss')
    #plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'taxi', '{} loss'.format(model_type)))
    plt.close()

    #print("Result of tensorflow_ai is {}".format(model.score(train_data_crosspart, train_label_crosspart)))
    print('tensorflow_ai evaluate:', model.evaluate(train_data_crosspart, train_label_crosspart))

    # Output
    predicted_fare_amount_list = []
    for predicted_fare_amount in model.predict(test_data):
        predicted_fare_amount_list.append(float(predicted_fare_amount))

    output = pd.DataFrame({'key':test_key, 'fare_amount': predicted_fare_amount_list})
    output.to_csv(os.path.join(os.getcwd(), 'taxi', 'tensorflow_ai_{}_submissions.csv'.format(model_type)), index=False)


if __name__ == '__main__':
    td, tl, tdt, tlt, tdc, tlc, test_data, test_key = taxi_data_deal()

    # package_ai
    ai_model_routing = {
        #'mlp' : MLPRegressor(random_state=1, hidden_layer_sizes=(400,1), max_iter=400), #too large
        'GReg' : GradientBoostingRegressor(random_state=0),
        'CAT' : CatBoostRegressor(verbose=0, loss_function='RMSE'), #iterations=10, learning_rate=0.03, loss_function='MAE', n_estimators=300, verbose=0
        'LGMB' : LGBMRegressor(),
        'XGBRegressor' : XGBRegressor(objective='reg:squarederror'),
        #'svr' : SVR(kernel='linear'), #too large
        'lasso' : Lasso(),
        'rf' : RandomForestRegressor(n_estimators=5, random_state=42),
    }
    for model_name, model in ai_model_routing.items():
        print(model_name)
        start = process_time()
        package_ai(td, tl, tdc, tlc, test_data, test_key, model, model_name)
        end = process_time()
        print('{} spent time:'.format(model_name), end-start)
        print()

    # tensorflow_ai
    start = process_time()
    tensorflow_ai(td, tl, tdc, tlc, test_data, model_type='dnn')
    end = process_time()
    print('tensorflow_ai spent time:', end-start)