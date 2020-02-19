import numpy as np
from Gev_network import MLP_AE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import os

import warnings

warnings.filterwarnings("ignore")


# fix random seed for reproducibility
seed = 150
np.random.seed(seed)


def one_hot_encoding(train, variable):
    values = train[variable].unique()
    if len(values)==2:
        train[str(values[0])] = np.where(train[variable] == values[0], 1, 0)
        train = train.drop(variable, axis=1)
    else:
        for val in values:
            train[str(val)] = np.where(train[variable]==val, 1, 0)
        train = train.drop(variable, axis=1)

    return train


def training (activation, loss_weight, data_name):

    res = {'activation': [], 'data': [], 'g-mean': [],
              'auc_score': [], 'acc': [], 'weight': [], 'brier': [], 'f-score': []}

    data = pd.read_csv('UCI/'  + data_name +'.csv')

    for var in list(data):
        if data[var].dtype == 'object':
            data = one_hot_encoding(data, var)

    data = np.asarray(data)

    length = data.shape[1] - 1
    X, Y = data[:, 0:length], data[:, length]

    scaler = MinMaxScaler()
    scaler = scaler.fit(X)
    scaled_X = scaler.transform(np.asarray(X))

    skf = StratifiedKFold(n_splits=5)

    data_index = 1

    for train_index, test_index in skf.split(scaled_X, Y):

        trainX, testX = scaled_X[train_index], scaled_X[test_index]
        trainY, testY = Y[train_index], Y[test_index]

        batch_size = 16


        for weight in loss_weight:
           model = MLP_AE(trainX=trainX, trainY=trainY, epoch_number=2000, batch_size=batch_size, learning_rate=0.001,
                          encoder=[32, 16, 8],decoder=[16, 32], sofnn=[32], early_stoppting_patience=200,
                          neurons=[32], activation=activation, reg_lambda=0.00001,
                          loss_weigth=weight, rand=data_index)

           final_model = model.MLP_AE('GEV_MODEL/UCI/model_ae_%s_%s' %(data_name, data_index),
                                      'GEV_MODEL/UCI/model_%s_%s_%s_%s.h5' %(data_name, activation, data_index, weight))

           pred_Y, true_Y = model.predict(testX, testY, final_model)

           brier_score, auc_score, f_score, acc, gmean = model.model_evaluation(pred_Y, true_Y)

           print("%s: 'Brier' %0.4f, 'AUC', %0.4f, 'F', %0.4f, 'ACC', %0.4f, "
                 "'Gmean', %0.4f" %(data_name, brier_score, auc_score, f_score, acc, gmean))

           res['activation'].append(activation)
           res['data'].append(data_name)
           res['g-mean'].append(gmean)
           res['auc_score'].append(auc_score)
           res['acc'].append(acc)
           res['f-score'].append(f_score)
           res['brier'].append(brier_score)
           res['weight'].append(weight)

           data_index = data_index + 1

    res_brier = pd.DataFrame.from_dict(res)

    res_brier.to_csv('GEV_MODEL/UCI/' + 'result_final.csv', mode='a',
                     encoding='euc-kr', index=False)

    return res


loss_weight= [0.05]


data_names = ['bupa', 'column_2C', 'ionosphere',
              'parkinsons', 'sonar']

for data_name in data_names:

    result_sigmoid = training(activation='sigmoid', loss_weight=loss_weight,
                              data_name=data_name)
    result_gev = training(activation='gev', loss_weight=loss_weight,
                          data_name=data_name)






