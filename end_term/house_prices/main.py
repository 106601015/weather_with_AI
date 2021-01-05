import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pretty_errors
import time

from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

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


def house_prices_data_deal():
    # Create train/test dataset(data:without SalePrice, label:SalePrice)
    try:
        test_data = pd.read_csv(os.path.join(os.getcwd(), 'test.csv'))
        train_data = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))
    except FileNotFoundError:
        test_data = pd.read_csv(os.path.join(os.getcwd(), 'house_prices', 'test.csv'))
        train_data = pd.read_csv(os.path.join(os.getcwd(), 'house_prices', 'train.csv'))
    train_label = train_data['SalePrice']
    train_data = train_data.drop('SalePrice', axis=1)

    # Cleaning test_data dataset's nullcolumns, and record in NANColumns list
    NANColumns = []
    i = 0
    for a in test_data.isnull().sum():
        # if this column has null values
        if a != 0:
            #print(test_data.columns[i], 'loss {} values'.format(a))
            NANColumns.append(test_data.columns[i])
        i += 1
    print()
    print('test_data have {} columns'.format(i))
    print('but {} columns have null values'.format(len(NANColumns)))

    # Replace train/test dataset nan values with mean/'X', and drop useless columns(data predeal)
    train_data["LotFrontage"]   =train_data["LotFrontage"].replace(np.nan, np.mean(train_data["LotFrontage"]))
    train_data["GarageYrBlt"]   =train_data["GarageYrBlt"].replace(np.nan, np.mean(train_data["GarageYrBlt"]))
    train_data["MasVnrArea"]    =train_data["MasVnrArea"].replace(np.nan, np.mean(train_data["MasVnrArea"]))
    train_data["BsmtQual"]      =train_data["BsmtQual"].replace(np.nan,"X")
    train_data["BsmtCond"]      =train_data["BsmtCond"].replace(np.nan,"X")
    train_data["BsmtExposure"]  =train_data["BsmtExposure"].replace(np.nan,"X")
    train_data["BsmtFinType1"]  =train_data["BsmtFinType1"].replace(np.nan,"X")
    train_data["BsmtFinType2"]  =train_data["BsmtFinType2"].replace(np.nan,"X")
    train_data["Electrical"]    =train_data["Electrical"].replace(np.nan,"X")
    train_data["FireplaceQu"]   =train_data["FireplaceQu"].replace(np.nan,"X")
    train_data["GarageType"]    =train_data["GarageType"].replace(np.nan,"X")
    train_data["GarageFinish"]  =train_data["GarageFinish"].replace(np.nan,"X")
    train_data["GarageQual"]    =train_data["GarageQual"].replace(np.nan,"X")
    train_data["GarageCond"]    =train_data["GarageCond"].replace(np.nan,"X")
    train_data=train_data.drop(columns=["Alley","PoolQC","MasVnrType","Fence","MiscFeature","Id"])

    test_data["LotFrontage"]    =test_data["LotFrontage"].replace(np.nan, np.mean(test_data["LotFrontage"]))
    test_data["GarageYrBlt"]    =test_data["GarageYrBlt"].replace(np.nan, np.mean(test_data["GarageYrBlt"]))
    test_data["BsmtFinSF1"]     =test_data["BsmtFinSF1"].replace(np.nan, np.mean(test_data["BsmtFinSF1"]))
    test_data["BsmtFinSF2"]     =test_data["BsmtFinSF2"].replace(np.nan, np.mean(test_data["BsmtFinSF2"]))
    test_data["BsmtUnfSF"]      =test_data["BsmtUnfSF"].replace(np.nan, np.mean(test_data["BsmtUnfSF"]))
    test_data["TotalBsmtSF"]    =test_data["TotalBsmtSF"].replace(np.nan, np.mean(test_data["TotalBsmtSF"]))
    test_data["BsmtHalfBath"]   =test_data["BsmtHalfBath"].replace(np.nan, np.mean(test_data["BsmtHalfBath"]))
    test_data["BsmtFullBath"]   =test_data["BsmtFullBath"].replace(np.nan, np.mean(test_data["BsmtFullBath"]))
    test_data["GarageArea"]     =test_data["GarageArea"].replace(np.nan, np.mean(test_data["GarageArea"]))
    test_data["GarageCars"]     =test_data["GarageCars"].replace(np.nan, np.mean(test_data["GarageCars"]))
    test_data["MasVnrArea"]     =test_data["MasVnrArea"].replace(np.nan, np.mean(test_data["MasVnrArea"]))
    test_data["BsmtQual"]       =test_data["BsmtQual"].replace(np.nan,"X")
    test_data["BsmtCond"]       =test_data["BsmtCond"].replace(np.nan,"X")
    test_data["BsmtExposure"]   =test_data["BsmtExposure"].replace(np.nan,"X")
    test_data["BsmtFinType1"]   =test_data["BsmtFinType1"].replace(np.nan,"X")
    test_data["BsmtFinType2"]   =test_data["BsmtFinType2"].replace(np.nan,"X")
    test_data["Electrical"]     =test_data["Electrical"].replace(np.nan,"X")
    test_data["FireplaceQu"]    =test_data["FireplaceQu"].replace(np.nan,"X")
    test_data["GarageType"]     =test_data["GarageType"].replace(np.nan,"X")
    test_data["GarageFinish"]   =test_data["GarageFinish"].replace(np.nan,"X")
    test_data["GarageQual"]     =test_data["GarageQual"].replace(np.nan,"X")
    test_data["GarageCond"]     =test_data["GarageCond"].replace(np.nan,"X")
    test_data["SaleType"]       =test_data["SaleType"].replace(np.nan,"o")
    test_data["Functional"]     =test_data["Functional"].replace(np.nan,"Typ")
    test_data["KitchenQual"]    =test_data["KitchenQual"].replace(np.nan,"Gd")
    test_data["MSZoning"]       =test_data["MSZoning"].replace(np.nan,"X")
    test_data["Utilities"]      =test_data["Utilities"].replace(np.nan,"X")
    test_data["Exterior1st"]    =test_data["Exterior1st"].replace(np.nan,"X")
    test_data["Exterior2nd"]    =test_data["Exterior2nd"].replace(np.nan,"X")
    test_data["GarageCond"]     =test_data["GarageCond"].replace(np.nan,"X")
    test_data=test_data.drop(columns=["Alley","PoolQC","MasVnrType","Fence","MiscFeature","Id"])

    # Split dataset and label, create complete/trainpart/crosspart data/label
    split_num = 1200
    all_num = 1460
    train_data_complete = train_data
    train_data_trainpart = train_data[0:split_num]
    train_data_crosspart = train_data[split_num:all_num] ############ train_data should train_data_trainpart
    train_label_complete = train_label
    train_label_trainpart = train_label[0:split_num]
    train_label_crosspart = train_label[split_num:all_num]

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

    return train_data_complete, train_label_complete, train_data_trainpart, train_label_trainpart, train_data_crosspart, train_label_crosspart, test_data

def package_ai(train_data_complete, train_label_complete, train_data_crosspart, train_label_crosspart, test_data, model, model_name):
    model.fit(train_data_complete, train_label_complete)
    print("Result of {} is {}".format(model_name, model.score(train_data_crosspart, train_label_crosspart)))

    # Output
    predicted_prices_list = []
    for predicted_prices in model.predict(test_data):
        predicted_prices_list.append(int(predicted_prices))

    output = pd.DataFrame({'Id':range(1461, 2920), 'SalePrice': predicted_prices_list})
    output.to_csv(os.path.join(os.getcwd(), 'house_prices', '{}.csv'.format(model_name)), index=False)

def tensorflow_ai(train_data_complete, train_label_complete, train_data_crosspart, train_label_crosspart, test_data):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
            units = 1,
            #input_shape = [1,74],
            kernel_initializer = 'ones',
            kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=1),
        )
    )
    model.add(tf.keras.layers.Dense(50))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mae')
    train_history = model.fit(train_data_complete.values, train_label_complete.values, batch_size=8, epochs=15)

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
    plt.show()

    #print("Result of tensorflow_ai is {}".format(model.score(train_data_crosspart, train_label_crosspart)))
    print('tensorflow_ai evaluate:', model.evaluate(train_data_crosspart.values, train_label_crosspart.values))

    # Output
    predicted_prices_list = []
    for predicted_prices in model.predict(test_data):
        predicted_prices_list.append(int(predicted_prices))

    output = pd.DataFrame({'Id':range(1461, 2920), 'SalePrice': predicted_prices_list})
    output.to_csv(os.path.join(os.getcwd(), 'house_prices', 'tensorflow_ai.csv'), index=False)


if __name__ == '__main__':
    td, tl, tdt, tlt, tdc, tlc, test_data = house_prices_data_deal()

    # package_ai
    '''
    ai_model_routing = {
        'mlp' : MLPRegressor(random_state=1,hidden_layer_sizes=(400,1),max_iter=400),
        'GReg' : GradientBoostingRegressor(random_state=0),
        'CAT' : CatBoostRegressor(verbose=0, n_estimators=300),
        'LGMB' : LGBMRegressor(),
        'XGBRegressor' : XGBRegressor(objective='reg:squarederror'),
    }
    for model_name, model in ai_model_routing.items():
        start = time.clock()
        package_ai(td, tl, tdc, tlc, test_data, model, model_name)
        end = time.clock()
        print('{} spent time:'.format(model_name), end-start)
        print()
    '''

    # tensorflow_ai
    start = time.clock()
    tensorflow_ai(td, tl, tdc, tlc, test_data)
    end = time.clock()
    print('tensorflow_ai spent time:', end-start)
