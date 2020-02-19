from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.initializers import *
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy.spatial.distance import euclidean
from keras.models import load_model
import os
from keras.optimizers import adam
from imblearn.metrics import geometric_mean_score
from keras.losses import mse, binary_crossentropy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

class Gev(Activation):

    def __init__(self, activation, **kwargs):
        super(Gev, self).__init__(activation, **kwargs)
        self.__name__ = 'gev'

def gev(x):
    return K.exp(-K.exp(-x))

get_custom_objects().update({'gev': Gev(gev)})


def distance_calc(tensors):
    euclid_dist = K.mean(K.square(tensors[0] - tensors[1]), axis=-1)
    euclid_dist = K.expand_dims(euclid_dist, 1)

    x1_val = K.sqrt(K.tf.reduce_sum(K.tf.matmul(tensors[0], K.transpose(tensors[0]))))
    x2_val = K.sqrt(K.tf.reduce_sum(K.tf.matmul(tensors[1], K.transpose(tensors[1]))))

    denom = K.tf.multiply(x1_val, x2_val)
    num = K.tf.reduce_sum(K.tf.multiply(tensors[0], tensors[1]), axis=1)

    cos_dist = K.tf.div(num, denom)
    cos_dist = K.expand_dims(cos_dist, 1)

    return [euclid_dist, cos_dist]

def distance_output_shape(input_shapes):
    shape1 = [input_shapes[0][0],1]
    shape2 = [input_shapes[1][0], 1]
    return [tuple(shape1), tuple(shape2)]

class AE:

    def __init__(self, trainX, valX, epoch_number, batch_size, learning_rate, encoder,
                 decoder, early_stoppting_patience, activation,
                 reg_lambda=0.0001, rand=0):
        if len(trainX.shape) < 2:
            trainX = np.expand_dims(trainX, axis=1)
        self.trainX = trainX
        self.valX = valX
        self.epoch = epoch_number
        self.batch = batch_size
        self.lr = learning_rate
        self.early_stopping = early_stoppting_patience
        self.encoder = encoder
        self.decoder = decoder
        self.array_size = trainX.shape[1]
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.rand = rand

    def AutoEncoder(self, model_path_ae):

        if os.path.isfile(model_path_ae+".h5"):

            decoder = load_model(model_path_ae + ".h5")
            encoder = Model(decoder.layers[0].input, decoder.layers[1].get_output_at(-1))

            return encoder, decoder

        else:

            trainX, valX = self.trainX, self.valX

            input_dim = self.array_size

            ### Auto-Encoder
            input_encoder = Input(shape=(input_dim,), name='input_encoder')

            encoded_output = Dense(self.encoder[0], activation='relu')(input_encoder)

            for layer in self.encoder[1:]:
                encoded_output = Dense(layer, activation='relu')(encoded_output)

            encoder = Model(input_encoder, encoded_output, name='encoder')

            input_dim_decoder = int(encoded_output.shape[1])
            input_decoder = Input(shape=(input_dim_decoder,), name='decoder_input')

            decoded_output = Dense(self.decoder[0], activation='relu')(input_decoder)

            for layer in self.decoder[1:]:
                decoded_output = Dense(layer, activation='relu')(decoded_output)

            decoded_output = Dense(self.array_size, activation='sigmoid')(decoded_output)
            decoder = Model(input_decoder, decoded_output, name='decoder_loss')

            outputs = decoder(encoder(input_encoder))

            ae = Model(input_encoder, outputs, name='AutoEncoder_loss')
            print(ae.summary())

            optimizer = adam(lr=self.lr, epsilon=None, decay=0, amsgrad=False)

            ae.compile(loss='mean_squared_error', optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=0, mode='auto'),
                          ModelCheckpoint(model_path_ae+".h5", monitor='val_loss', verbose=0, save_best_only=True)]

            ae.fit(trainX, trainX, epochs=self.epoch, batch_size=self.batch, verbose=2,
                            shuffle=False, validation_data=(valX, valX), callbacks=early_stop)

            decoder = load_model(model_path_ae+".h5")
            encoder = Model(decoder.layers[0].input, decoder.layers[1].get_output_at(-1))

            return encoder, decoder


class MLP_AE:

    def __init__(self, trainX, trainY, epoch_number, batch_size, learning_rate, encoder,
                 decoder, sofnn, early_stoppting_patience, neurons, activation,
                 reg_lambda=0.0001, loss_weigth=0.25, rand=0):

        if len(trainX.shape)<2:
            trainX = np.expand_dims(trainX, axis=1)
        self.trainX = trainX
        self.trainY = trainY
        self.epoch = epoch_number
        self.batch = batch_size
        self.lr = learning_rate
        self.early_stopping = early_stoppting_patience
        self.neurons = neurons
        self.encoder = encoder
        self.decoder = decoder
        self.sofnn = sofnn
        self.array_size = trainX.shape[1]
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.loss_weigth = loss_weigth
        self.rand=rand


    def data_preprocessing(self):

        trainX, valX, trainY, valY = train_test_split(self.trainX, self.trainY, test_size=0.20,
                                                        random_state=12345)

        return trainX, valX, trainY, valY

    def MLP_AE(self, model_path_ae, model_path):

        trainX, valX, trainY, valY = self.data_preprocessing()


        if os.path.isfile(model_path):

            final_model = load_model(model_path)

        else:
            input_dim = self.array_size
            input_importance =Input(shape=(input_dim,), name='importance_input')
            input_encoder = Input(shape=(input_dim,), name='input_encoder')

            ### input selection neural network
            Input_selection = Sequential()
            Input_selection.add(Dense(self.sofnn[0], activation='tanh',
                                     input_dim=input_dim))

            if len(self.sofnn) > 1:
                for layer_size in self.sofnn[1:]:
                    Input_selection.add(Dense(layer_size, activation='tanh'))

            Input_selection.add((Dense(self.array_size, activation='softmax')))

            variable_importance = Input_selection(input_importance)
            importance = Model(input_importance, variable_importance, name='importance')

            selected_input = Multiply()([importance(input_importance), input_importance])

            ## auto_encoder

            ae = AE(trainX=self.trainX, valX= valX, epoch_number=self.epoch, batch_size=self.batch, learning_rate=self.lr,
                    encoder=self.encoder, decoder=self.decoder, early_stoppting_patience=self.early_stopping,
                    activation=self.activation, reg_lambda=self.reg_lambda, rand=self.rand)

            encoder, decoder = ae.AutoEncoder(model_path_ae)

            outputs = decoder(input_encoder)

            layer = Lambda(distance_calc, distance_output_shape)
            euclid_dist, cos_dist = layer([input_encoder, outputs])

            final_input = Concatenate()([euclid_dist, cos_dist, selected_input, encoder(input_encoder)])

            input_dim_mlp = int(final_input.shape[1])
            input_mlp = Input(shape=(input_dim_mlp,), name='input_mlp')

            Prediction = Dense(self.neurons[0], activation=self.activation,
                               kernel_regularizer=regularizers.l1(self.reg_lambda))(input_mlp)
            if len(self.neurons) > 1:
                for layer_size in self.neurons[1:]:
                    Prediction=Dense(layer_size, activation=self.activation,
                                     kernel_regularizer=regularizers.l1(self.reg_lambda))(Prediction)

            predicted_output = Dense(1, activation=self.activation)(Prediction)

            Prediction_MLP = Model(input_mlp, predicted_output, name='Prediction_loss')
            predicted_output = Prediction_MLP(final_input)

            final_model = Model(inputs=[input_importance, input_encoder], outputs=[outputs, predicted_output])
            print(final_model.summary())

            losses = {'AutoEncoder_loss': 'mean_squared_error',
                      'Prediction_loss': 'binary_crossentropy'}

            lossWeights = {"AutoEncoder_loss": self.loss_weigth, "Prediction_loss": 1}


            optimizer = adam(lr=self.lr, epsilon=None, decay=0, amsgrad=False)

            final_model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=0, mode='auto'),
                          ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True)]

            final_model.fit([trainX, trainX], [trainX, trainY], epochs=self.epoch, batch_size=self.batch, verbose=2,
                           shuffle=False, validation_data=([valX, valX], [valX, valY]), callbacks=early_stop)

        final_model = load_model(model_path)

        return final_model

    def predict(self, testX, testY, final_model):

        _, pred_Y = final_model.predict([testX, testX])

        return pred_Y, testY

    def model_evaluation(self, pred_Y, true_Y):


        brier_score = brier_score_loss(true_Y, pred_Y)
        auc_score = roc_auc_score(true_Y, pred_Y)

        fpr, tpr, thresholds = roc_curve(true_Y, pred_Y)

        best = [0, 1]
        dist = []
        for (x, y) in zip(fpr, tpr):
            dist.append([euclidean([x, y], best)])

        bestPoint = [fpr[dist.index(min(dist))], tpr[dist.index(min(dist))]]
        bestCutOff = thresholds[list(tpr).index(bestPoint[1])]

        pred_Y = np.where(pred_Y >= bestCutOff, 1, 0)
        acc = accuracy_score(true_Y, pred_Y)
        f_score = f1_score(true_Y, pred_Y, average='weighted')

        gmean = geometric_mean_score(true_Y, pred_Y, average='weighted')

        return brier_score, auc_score, f_score, acc, gmean

