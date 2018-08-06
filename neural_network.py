import numpy as np
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
import keras.backend as K
from time import time


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# TODO: Change early stopping when validation is absent
class FNNModel(BaseEstimator):
    def __init__(self, hidden_layers, dropout, activation, optimizer, metrics,
                 loss, epochs, batch_size, timeit, verbosity, callbacks,
                 class_weight, validation_split, validation_data, early_stopping,
                 learning_rate):

        self.hidden_layers = list(hidden_layers)

        if isinstance(dropout, list):
            assert len(hidden_layers) == len(dropout)
            self.dropout = dropout
        else:
            self.dropout = [dropout] * len(hidden_layers)
        self.activation = activation

        self.learning_rate = learning_rate
        if learning_rate == 'auto':
            self.optimizer = optimizer
        else:
            if optimizer == 'adam':
                self.optimizer = Adam(lr=learning_rate)
            elif optimizer == 'sgd':
                self.optimizer = SGD(lr=learning_rate)
            elif optimizer == 'rmsprop':
                self.optimizer = RMSprop(lr=learning_rate)
            else:
                raise ValueError(f"'{optimizer} is not supported with user defined learning rate. "
                                 f"Use 'adam', 'sgd' or 'rmsprop'")

        self.early_stopping = int(early_stopping)

        if self.early_stopping > 0:
            callbacks += (EarlyStopping(patience=(self.early_stopping - 1)),)

        self.optimizer = optimizer
        self.metrics = list(metrics)
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.timeit = timeit
        self.verbosity = verbosity
        self.callbacks = list(callbacks)
        self.class_weight = class_weight
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.model = Sequential()
        self.label_binarizer = None

    def summary(self):
        return self.model.summary()


class FNNClassifier(FNNModel):
    def __init__(self, hidden_layers=(50,), dropout=0.5, activation='softmax', optimizer='adam', metrics=('accuracy',),
                 loss='categorical_crossentropy', epochs=100, batch_size=128, timeit=True, verbosity=2, callbacks=(),
                 class_weight=None, validation_split=0.1, validation_data=None, early_stopping=5,
                 learning_rate='auto'):
        super().__init__(hidden_layers=hidden_layers, dropout=dropout, activation=activation, optimizer=optimizer,
                         metrics=metrics,
                         loss=loss, epochs=epochs, batch_size=batch_size, timeit=timeit, verbosity=verbosity,
                         callbacks=callbacks,
                         class_weight=class_weight, validation_split=validation_split, validation_data=validation_data,
                         early_stopping=early_stopping,
                         learning_rate=learning_rate)

    def fit(self, X, y):
        if self.timeit:
            start_time = time()
        n_features = X.shape[1]

        if self.verbosity:
            print(
                'Data size ({:d}, {:d}) -\t Epochs {:d} -\t Batch Size {:d}'.format(X.shape[0], X.shape[1], self.epochs,
                                                                                    self.batch_size))
        if len(y.shape) == 1 and len(np.unique(y)) > 2:
            self.label_binarizer = LabelBinarizer()
            y = self.label_binarizer.fit_transform(y)

        # TODO: Add error cases like multiple labels
        if len(y.shape) == 2:
            n_classes = y.shape[1]
            print(n_classes)
            if self.class_weight == 'balanced':
                weights = list(y.shape[0] / (n_classes * y.sum(axis=0)))
                self.class_weight = {i: weights[i] for i in range(len(weights))}
                print('Computed Class Weights', self.class_weight)
        elif len(y.shape) == 1:
            n_classes = 1
            self.loss = 'binary_crossentropy'
            self.activation = 'sigmoid'
            if self.class_weight == 'balanced':
                weights = list(y.shape[0] / (2 * np.bincount(y)))
                self.class_weight = {0: weights[0], 1: weights[1]}
                print('Computed Class Weights', self.class_weight)
        else:
            raise ValueError("Invalid Label")

        K.clear_session()
        self.model.add(InputLayer(input_shape=(n_features,), name='Input'))
        for i, h, d in zip(range(0, len(self.hidden_layers)), self.hidden_layers, self.dropout):
            self.model.add(Dense(units=h, activation='relu', name=f'Hidden_{i+1}'))
            if d > 0:
                self.model.add(Dropout(d, name=f'Dropout_{i+1}_{d}'))
        self.model.add(Dense(units=n_classes, activation=self.activation, name=f'Output_{self.activation}'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if self.verbosity == 2:
            self.model.summary()
        self.model.fit(X, y, verbose=(self.verbosity - 1), epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks, class_weight=self.class_weight, validation_data=self.validation_data,
                       validation_split=self.validation_split)
        if self.timeit:
            print('Fit complete in {:.2f} seconds'.format(time() - start_time))

    def predict(self, X):
        return np.array(self.model.predict(X, batch_size=self.batch_size) > 0.5, dtype=np.uint8)

    def predict_proba(self, X):
        if self.activation == 'sigmoid':
            return np.hstack([1 - self.model.predict(X, batch_size=self.batch_size),
                              self.model.predict(X, batch_size=self.batch_size)])
        else:
            return self.model.predict(X, batch_size=self.batch_size)

    def score(self, X, y):
        if self.label_binarizer is not None:
            y = self.label_binarizer.fit_transform(y)
        score = self.model.evaluate(x=X, y=y)
        return {"accuracy": score[1], "loss": score[0]}


class FNNRegressor(FNNModel):
    def __init__(self, hidden_layers=(50,), dropout=0.5, activation='linear', optimizer='adam', metrics=('mse',),
                 loss='mse', epochs=100, batch_size=128, timeit=True, verbosity=2, callbacks=(),
                 class_weight=None, validation_split=0.1, validation_data=None, early_stopping=5,
                 learning_rate='auto'):
        super().__init__(hidden_layers=hidden_layers, dropout=dropout, activation=activation, optimizer=optimizer,
                         metrics=metrics, loss=loss, epochs=epochs, batch_size=batch_size, timeit=timeit,
                         verbosity=verbosity, callbacks=callbacks, class_weight=class_weight,
                         validation_split=validation_split, validation_data=validation_data,
                         early_stopping=early_stopping, learning_rate=learning_rate)

    def fit(self, X, y):
        if self.timeit:
            start_time = time()
        n_features = X.shape[1]

        if self.verbosity:
            print(
                'Data size ({:d}, {:d}) -\t Epochs {:d} -\t Batch Size {:d}'.format(X.shape[0], X.shape[1], self.epochs,
                                                                                    self.batch_size))
        K.clear_session()
        self.model.add(InputLayer(input_shape=(n_features,), name='Input'))
        for i, h, d in zip(range(0, len(self.hidden_layers)), self.hidden_layers, self.dropout):
            self.model.add(Dense(units=h, activation='relu', kernel_initializer='normal', name=f'Hidden_{i+1}'))
            if d > 0:
                self.model.add(Dropout(d, name=f'Dropout_{i+1}_{d}'))
        self.model.add(Dense(units=1, activation=self.activation, kernel_initializer='normal', name=f'Output_{self.activation}'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if self.verbosity == 2:
            self.model.summary()
        self.model.fit(X, y, verbose=(self.verbosity - 1), epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks, class_weight=self.class_weight, validation_data=self.validation_data,
                       validation_split=self.validation_split)
        if self.timeit:
            print('Fit complete in {:.2f} seconds'.format(time() - start_time))

    def predict(self, X):
        return np.array(self.model.predict(X, batch_size=self.batch_size), dtype=np.float64)

    def score(self, X, y):
        return self.model.evaluate(x=X, y=y)

