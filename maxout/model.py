from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, MaxoutDense
from helpers import EarlyStopping
from hp_toolkit.hp import Model, Param, make_constant_param
from keras.optimizers import SGD, Adam  # NOQA

optimization_params = dict(
        lr=make_constant_param(0.01),
        decay=make_constant_param(1E-6),
        momentum=make_constant_param(0.9),
        patience=make_constant_param(20),
)


class Base(Model):
    def fit(self, X, y):
        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=self.patience,
                                       smaller_is_better=False)
        validation_data = (self.validation_data
                           if hasattr(self, "validation_data") else None)
        self.model.fit(
            X, y,
            nb_epoch=100,
            show_accuracy=True,
            validation_data=validation_data,
            callbacks=[early_stopping])

    def predict(self, X):
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def prepare_optimization_(self, model):
        sgd = SGD(lr=self.lr, decay=self.decay,
                  momentum=self.momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model


class DenseModel(Base):
    params = dict(
            units1=Param(initial=100, interval=[100, 800], type='int'),
            units2=Param(initial=100, interval=[100, 800], type='int'),
            feats1=Param(initial=5, interval=[5, 20], type='int'),
            feats2=Param(initial=5, interval=[5, 20], type='int'),
            dropout1=Param(initial=0.5,
                           interval=[0.1, 0.3, 0.5, 0.7, 0.9], type='choice'),
            dropout2=Param(initial=0.5,
                           interval=[0.1, 0.3, 0.5, 0.7, 0.9], type='choice'),
    )
    params.update(optimization_params)

    def build(self, nb_features=784, nb_outputs=10):
        model = Sequential()
        model.add(MaxoutDense(self.units1, input_dim=nb_features,
                              nb_feature=self.feats1))
        model.add(Dropout(self.dropout1))
        model.add(MaxoutDense(self.units2, nb_feature=self.feats2))
        model.add(Dropout(self.dropout2))
        model.add(Dense(nb_outputs))
        model.add(Activation('softmax'))
        model = self.prepare_optimization_(model)
        self.model = model
        return model
