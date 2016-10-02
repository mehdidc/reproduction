from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, MaxoutDense, Lambda, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from helpers import EarlyStopping
from hp_toolkit.hp import Model, Param, make_constant_param
from keras.optimizers import SGD, Adam, RMSprop  # NOQA
from keras.layers.advanced_activations import LeakyReLU, PReLU

optimization_params = dict(
        #lr=make_constant_param(0.001),
        lr=Param(initial=0.001, interval=[-4, -2], type='real', scale='log10'),
        decay=Param(initial=0.98, interval=[0.8, 1], type='real'),
        momentum=Param(initial=0.9, interval=[0.5, 0.99], type='real'),
        patience=make_constant_param(50),
        val_ratio=make_constant_param(0.15),
        max_epochs=make_constant_param(200),
        batch_size=Param(initial=128, interval=[16, 32, 64, 128, 256, 512], type='choice'),
)


class Base(Model):
    def fit(self, X, y, **kwargs):
        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=self.patience,
                                       smaller_is_better=False)
        validation_data = (self.validation_data
                           if hasattr(self, "validation_data") else None)
        if validation_data is None:
            kwargs["validation_split"] = self.val_ratio
        else:
            kwargs["validation_data"] = validation_data
        #if "callbacks" in kwargs:
        #    kwargs["callbacks"].append(early_stopping)
        #else:
        #    kwargs["callbacks"] = [early_stopping]
        return self.model.fit(X, y,
                              nb_epoch=self.max_epochs,
                              batch_size=self.batch_size,
                              show_accuracy=True,
                              **kwargs)

    def predict(self, X):
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def prepare_optimization_(self, model):
        sgd = Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model


class DenseModel(Base):
    params = dict(
            units1=Param(initial=200, interval=[100, 1500], type='int'),
            units2=Param(initial=200, interval=[100, 1500], type='int'),
            feats1=Param(initial=3, interval=[5, 20], type='int'),
            feats2=Param(initial=3, interval=[5, 20], type='int'),
            dropout1=Param(initial=0.2,
                           interval=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                           type='choice'),
            dropout2=Param(initial=0.1,
                           interval=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                           type='choice'),
    )
    params.update(optimization_params)

    def build(self, nb_features=784, nb_outputs=10, imshape=None):
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


class ConvModel(Base):

    params = dict(
        nb_filters1=Param(initial=96, interval=[48, 120], type='int'),
        nb_filters2=Param(initial=96, interval=[48, 120], type='int'),
        nb_filters3=Param(initial=48, interval=[24, 84], type='int'),
        conv1=Param(initial=5, interval=[5], type='choice'),
        conv2=Param(initial=5, interval=[5], type='choice'),
        conv3=Param(initial=5, interval=[5], type='choice'),
        poolsize1=Param(initial=2, interval=[4], type='choice'),
        poolsize2=Param(initial=2, interval=[4], type='choice'),
        poolsize3=Param(initial=2, interval=[2], type='choice'),
        poolstride1=Param(initial=2, interval=[2], type='choice'),
        poolstride2=Param(initial=2, interval=[2], type='choice'),
        poolstride3=Param(initial=2, interval=[2], type='choice'),
        nbpieces1=Param(initial=2, interval=[2, 10], type='choice'),
        nbpieces2=Param(initial=2, interval=[2, 10], type='choice'),
        nbpieces3=Param(initial=4, interval=[4, 10], type='choice'),
    )
    params.update(optimization_params)

    def build(self, nb_features=784, nb_outputs=10, imshape=None):
        assert imshape is not None
        def conv_maxout_(X):
            return K.max(X, axis=2)
        def conv_maxout(model, nb_pieces, pad, *args, **kwargs):
            conv = Convolution2D(*args, **kwargs)
            model.add(conv)
            if pad > 0:
                padding = ZeroPadding2D(padding=(pad, pad))
                model.add(padding)
                conv = padding
            shape = conv.output_shape[1:]
            new_shape = (shape[0] / nb_pieces, nb_pieces, shape[1], shape[2])
            model.add(Reshape(new_shape))
            new_shape = (shape[0] / nb_pieces, shape[1], shape[2])
            l = Lambda(conv_maxout_, output_shape=new_shape)
            model.add(l)
            print(l.output_shape)
        model = Sequential()
        model.add(Reshape(imshape, input_shape=[nb_features]))
        conv_maxout(model,
                    self.nbpieces1,
                    0,
                    self.nb_filters1 * self.nbpieces1,
                    self.conv1, self.conv1,
                    activation='relu',
                    input_shape=imshape)
        #model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(self.poolsize1, self.poolsize1),
                  strides=(self.poolstride1, self.poolstride1)))
        conv_maxout(
             model,
             self.nbpieces2,
             3,
             self.nb_filters2 * self.nbpieces2,
             self.conv2, self.conv2,
             activation='relu'
        )
        #model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(self.poolsize2, self.poolsize2),
                 strides=(self.poolstride2, self.poolstride2)))
        conv_maxout(
             model,
             self.nbpieces3,
             3,
             self.nb_filters3 * self.nbpieces3,
             self.conv3, self.conv3,
             activation='relu'
        )
        #model.add(BatchNormalization())

        model.add(Dropout(0.5))

        model.add(MaxPooling2D(pool_size=(self.poolsize3, self.poolsize3),
                  strides=(self.poolstride3, self.poolstride3)))
        model.add(Flatten())
        model.add(Dense(nb_outputs))
        model.add(Activation('softmax'))
        model = self.prepare_optimization_(model)
        self.model = model
        return model


class Bogdan(Base):
    params = dict(
        nb_filters_init=Param(initial=32, interval=[8, 64], type='int'),
        nb_units=Param(initial=625, interval=[100, 800], type='int')
    )
    params.update(optimization_params)

    def build(self, nb_features=784, nb_outputs=10, imshape=None):
        assert imshape is not None
        #init = 'he_normal'
        init = 'glorot_uniform'
        model = Sequential()
        model.add(Reshape(imshape, input_shape=[nb_features]))
        model.add(ZeroPadding2D(padding=(1, 1)))

        k = self.nb_filters_init
        model.add(Convolution2D(k, 3, 3, init=init))
        #model.add(BatchNormalization())
        #model.add(Activation(relu))
        model.add(PReLU())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(k * 2, 3, 3, init=init))
        #model.add(BatchNormalization())
        #model.add(Activation(relu))
        model.add(PReLU())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(k * 2 * 2, 3, 3, init=init))
        #model.add(BatchNormalization())
        #model.add(Activation(relu))
        model.add(PReLU())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(self.nb_units, init=init))
        #model.add(BatchNormalization())
        #model.add(Activation(relu))
        model.add(PReLU())
        model.add(Dropout(0.5))

        model.add(Dense(nb_outputs, activation='softmax'))
        model = self.prepare_optimization_(model)
        self.model = model
        return model
