import numpy as np
from Gev_network import MLP_AE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.utils.vis_utils import plot_model


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import warnings

warnings.filterwarnings("ignore")


# fix random seed for reproducibility
seed = 150
np.random.seed(seed)


def one_hot_encoding(train, test, variable):
    values = train[variable].unique()
    if len(values)==2:
        train[str(values[0])] = np.where(train[variable] == values[0], 1, 0)
        train = train.drop(variable, axis=1)
        test[str(values[0])] = np.where(test[variable] == values[0], 1, 0)
        test = test.drop(variable, axis=1)
    else:
        for val in values:
            train[str(val)] = np.where(train[variable]==val, 1, 0)
            test[str(val)] = np.where(test[variable] == val, 1, 0)
        train = train.drop(variable, axis=1)
        test = test.drop(variable, axis=1)

    return train, test


def training (activation, loss_weight, data_name, folder):

    res = {'activation': [], 'data': [], 'g-mean': [],
              'auc_score': [], 'acc': [], 'weight': [], 'brier': [], 'f-score': []}

    for data_index in range(1, 6):
        if folder == 'imb_IRhigherThan9p3':
            train = np.genfromtxt(
                'KEEL/%s/' % folder + data_name + '-fold' + '/%s-%stra.dat' % (data_name, data_index),
                comments='@', dtype=None, encoding='utf-8', delimiter=',', unpack=False)

            train = pd.DataFrame(train)

            test = np.genfromtxt(
                'KEEL/%s/' % folder + data_name + '-fold' + '/%s-%stst.dat' % (data_name, data_index),
                comments='@', dtype=None, encoding='utf-8', delimiter=',', unpack=False)

            test = pd.DataFrame(test)
        else:
            train = np.genfromtxt('KEEL/%s/' %folder + data_name + '/%s-5-%stra.dat' % (data_name, data_index),
                                comments='@', dtype=None, encoding='utf-8', delimiter=',', unpack=False)

            train = pd.DataFrame(train)

            test = np.genfromtxt('KEEL/%s/' %folder + data_name + '/%s-5-%stst.dat' % (data_name, data_index),
                                  comments='@', dtype=None, encoding='utf-8', delimiter=',', unpack=False)

            test = pd.DataFrame(test)

        if len(train.iloc[:,0]) < 500:
            batch_size = 16
        elif len(train.iloc[:,0]) < 1000:
            batch_size = 32
        elif len(train.iloc[:,0]) < 5000:
            batch_size = 64
        elif len(train.iloc[:,0]) < 10000:
            batch_size = 128
        elif len(train.iloc[:,0]) < 20000:
            batch_size = 256
        else:
            batch_size = 512

        for var in list(train):
            if train[var].dtype == 'object':
                train, test = one_hot_encoding(train, test, var)

        scaler = MinMaxScaler()
        scaler = scaler.fit(train)
        scaled_train = scaler.transform(np.asarray(train))
        scaled_test = scaler.transform(np.asarray(test))

        length = scaled_train.shape[1]-1
        trainX, trainY = scaled_train[:, 0:length], scaled_train[:, length]
        testX, testY = scaled_test[:, 0:length], scaled_test[:, length]

        for weight in loss_weight:
           model = MLP_AE(trainX=trainX, trainY=trainY, epoch_number=2000, batch_size=batch_size, learning_rate=0.001,
                          encoder=[32, 16, 8],decoder=[16, 32], sofnn=[32], early_stoppting_patience=200,
                          neurons=[32], activation=activation, reg_lambda=0.00001,
                          loss_weigth=weight, rand=data_index)

           final_model = model.MLP_AE('GEV_MODEL/%s/model_ae_%s_%s' %(folder, data_name, data_index),
                                      'GEV_MODEL/%s/model_%s_%s_%s_%s.h5' %(folder, data_name, activation, data_index, weight))

           plot_model(final_model, to_file='model_plot.png')
           exit('bye')
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

    res_brier = pd.DataFrame.from_dict(res)

    res_brier.to_csv('GEV_MODEL/' + 'result_final.csv', mode='a',
                     encoding='euc-kr', index=False)

    return res


loss_weight= [0.05]


names = pd.read_csv('data_names.csv')
data_names = names.loc[58:60]

for data_name, folder in zip(data_names['name'], data_names['folder']):

    result_sigmoid = training(activation='sigmoid', loss_weight=loss_weight,
                              data_name=data_name, folder=folder)
    result_gev = training(activation='gev', loss_weight=loss_weight,
                          data_name=data_name, folder=folder)






