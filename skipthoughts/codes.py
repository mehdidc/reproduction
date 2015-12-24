import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import numpy as np
from sklearn.base import BaseEstimator
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers, nonlinearities, updates, init, objectives
from nolearn.lasagne.handlers import EarlyStopping
from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1

lambda_regularization = 0.04

def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden3"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

def build_model(hyper_parameters):
    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 64, 64), # 3 = depth of input layer (color), 64x64 image
    use_label_encoder=True,
    verbose=1,
    # objective function
    objective=objective_with_L2,
    **hyper_parameters
    )  
    return net
 
hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    hidden3_num_units=200,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    max_epochs=200,
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)
 
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=32, conv1_filter_size=(5, 5), 
    pool1_pool_size=(3, 3),
    conv2_num_filters=16, conv2_filter_size=(3, 3), 
    pool2_pool_size=(2, 2),
    conv3_num_filters=32, conv3_filter_size=(2, 2), 
    pool3_pool_size=(1, 1),
    hidden4_num_units=500, hidden4_nonlinearity = nonlinearities.leaky_rectify, 
    #hidden4_regularization = regularization.l2,
    hidden5_num_units=500, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    #hidden5_regularization = regularization.l2,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.04,
    update_momentum=0.9,
    max_epochs=20,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_loss')]
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
 
class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        self.clf = RandomForestClassifier(
            n_estimators=10, max_features=2, max_leaf_nodes=5)
        self.clf.fit(X_vectorized, y)
 
    def predict(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        return self.clf.predict(X_vectorized)
 
    def predict_proba(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        return self.clf.predict_proba(X_vectorized)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        self.clf = RandomForestClassifier(
            n_estimators=10, max_features=2, max_leaf_nodes=5)
        self.clf.fit(X_vectorized, y)

    def predict(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        return self.clf.predict(X_vectorized)

    def predict_proba(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        return self.clf.predict_proba(X_vectorized)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class Classifier(BaseEstimator):
	def __init__(self):
	    pass

	def fit(self, X, y):
	    X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
	    self.clf = RandomForestClassifier(
		n_estimators=10, max_features=2, max_leaf_nodes=6)
	    self.clf.fit(X_vectorized, y)

	def predict(self, X):
	    X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
	    return self.clf.predict(X_vectorized)

	def predict_proba(self, X):
	    X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
	    return self.clf.predict_proba(X_vectorized)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

import math
from skimage import data
from skimage import transform as tf

from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1
 
lambda_regularization = 0.04
 
def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden5"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        return Xb, yb


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        # additional static transformations
        print 'start preprocess'
        tformslist = list()
        tformslist.append(tf.SimilarityTransform(scale=1))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = math.pi/10))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = -math.pi/10))

        X_new = np.zeros((X.shape[0] * 3, X.shape[1], X.shape[2], X.shape[3]))
        print 'X shape ', X.shape[0]
        for i in xrange(X.shape[0]):
            Xbase = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
            Xbase[10:54,10:54,:] = X[i,10:54,10:54,:]
            if i % 1000 == 0:
                print 'performed first ' + str((i)) + ' transformations.'
            for j in xrange(len(tformslist)):
                X_new[len(tformslist)*i + j, :, :, :] = tf.warp(Xbase, tformslist[j])
        print 'end preprocess'
        X_new = (X_new[:,10:54,10:54,:] / 255.)
        X_new = X_new.astype(np.float32)
        X_new = X_new.transpose((0, 3, 1, 2))
        return X_new
    
    def preprocess_test(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        y_new = np.zeros((y.shape[0] * 3))
        for i in xrange(y.shape[0]):
            for j in xrange(3):
                y_new[3*i + j] = y[i]
        return y_new.astype(np.int32)

    def fit(self, X, y):
        X_new = self.preprocess(X)
        self.net.fit(X_new, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess_test(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess_test(X)
        return self.net.predict_proba(X)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        #objective=objective_with_L2,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.very_leaky_rectify,
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.leaky_rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
from itertools import repeat


def sample_from_rotation_x( x ):
    x_extends = []
    x_extends_extend = x_extends.extend
    rot90 = np.rot90
    np_array = np.array
    for i in range(x.shape[0]):
        x_extends.extend([
        np_array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np_array([rot90(x[i,:,:,0]),rot90(x[i,:,:,1]), rot90(x[i,:,:,2])]),
        np_array([rot90(x[i,:,:,0],2),rot90(x[i,:,:,1],2), rot90(x[i,:,:,2],2)]),
        np_array([rot90(x[i,:,:,0],3),rot90(x[i,:,:,1],3), rot90(x[i,:,:,2],3)])
        ])
    return np_array(x_extends)
 
def sample_from_rotation_y(y):
    y_extends = []
    y_extends_extend = y_extends.extend
    for i in y:
        y_extends_extend( repeat( i ,4) )
    return np.array(y_extends)


 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb
 
    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        #X = X.transpose((0, 3, 1, 2))
        X = sample_from_rotation_x( X )        
        return X
 
    def predict_preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        #X = sample_from_rotation_x( X )        
        return X

    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self
 
    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('hidden7', layers.DenseLayer),
            ('hidden8', layers.DenseLayer),
            ('hidden9', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
 
hyper_parameters = dict(
    conv1_num_filters=128, conv1_filter_size=(2, 2), pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(1, 1), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(1, 1), pool3_pool_size=(2, 2),
    conv4_num_filters=64 , conv4_filter_size=(2, 2), pool4_pool_size=(4, 4),
    hidden4_num_units=1024,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=512,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    hidden6_num_units=512,
    hidden6_nonlinearity=nonlinearities.very_leaky_rectify,
    hidden6_W=init.GlorotUniform(gain='relu'),
    hidden7_num_units=256,
    hidden7_nonlinearity=nonlinearities.leaky_rectify,
    hidden7_W=init.GlorotUniform(gain='relu'),
    hidden8_num_units=128,
    hidden8_nonlinearity=nonlinearities.rectify,
    hidden8_W=init.GlorotUniform(gain='relu'),
    hidden9_num_units=64,
    hidden9_nonlinearity=nonlinearities.tanh,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, objectives, updates, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1
from nolearn.lasagne.handlers import EarlyStopping
from skimage import data
from skimage import transform

lambda_regularization = 1e-6

def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden4"], layers["hidden5"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, ::-1, :]
        X_tmp1 = Xb[indices, :, ::-1, :]
        Y_tmp1 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, :, ::-1]
        X_tmp2 = Xb[indices, :, :, ::-1]
        Y_tmp2 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp3 = Xb[indices, :, :, :]
        Y_tmp3 = yb[indices]    
        X_tmp3 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp4 = Xb[indices, :, :, ::-1]
        Y_tmp4 = yb[indices]    
        X_tmp4 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp5 = Xb[indices, :, ::-1, :]
        Y_tmp5 = yb[indices]    
        X_tmp5 = X_tmp3.transpose((0,1,3,2))
        
        Xb = np.append(Xb,X_tmp1,axis=0)
        Xb = np.append(Xb,X_tmp2,axis=0)
        Xb = np.append(Xb,X_tmp3,axis=0)
        Xb = np.append(Xb,X_tmp4,axis=0)
        Xb = np.append(Xb,X_tmp5,axis=0)
        yb = np.append(yb,Y_tmp1)
        yb = np.append(yb,Y_tmp2)
        yb = np.append(yb,Y_tmp3)
        yb = np.append(yb,Y_tmp4)
        yb = np.append(yb,Y_tmp5)
        
        # small rotation of the images
        lx = 44
        pad_lx = 64
        shift_x = lx/2.
        shift_y = lx/2.
        
        
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp6 = Xb[indices, :, ::-1, :]
        X_tmp6 = X_tmp6.transpose(0,2,3,1)
        X_tmp6 = np.pad(X_tmp6,((0,0),(10,10),(10,10),(0,0)),'constant', constant_values=(0,0))
        Y_tmp6 = yb[indices]
        x_rot = X_tmp6[0]
        x_rot = x_rot.reshape(1,pad_lx,pad_lx,3)
        
        
        # tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(15))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        
        for i in X_tmp6[1::]:
            tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(np.random.randint(30)-15))
            xdel = transform.warp(i, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)            
            xdel=xdel.reshape(1,pad_lx,pad_lx,3)
            x_rot=np.append(x_rot,xdel,axis=0)
        
        x_rot = x_rot[:, 10:54, 10:54, :]
        x_rot = x_rot.transpose(0,3,1,2)
        x_rot = x_rot.astype(np.float32)
        Xb = np.append(Xb,x_rot,axis=0)
        yb = np.append(yb,Y_tmp6)
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            # ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(4, 4), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(4, 4), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden4_nonlinearity = nonlinearities.leaky_rectify,
    # dropout4_p=0.3,
    #hidden4_regularization = lasagne.regularization.l2(hidden4),
    hidden5_num_units=500, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.3,
    #hidden5_regularization = regularization.l2,
    output_num_units=18, 
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    objective=objective_with_L2,
    update=updates.adagrad,
    max_epochs=150,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=40, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=150)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        
        
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500, hidden5_num_units=500,
    dropout5_p=0.5,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=256)
)


class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping
import skimage.color
import skimage.transform

n = 10

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),

            ('conv1', layers.Conv2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('conv5', layers.Conv2DLayer),

            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),

            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64-(2*n), 64-(2*n)),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=16, conv1_filter_size=(3, 3),

    conv2_num_filters=32, conv2_filter_size=(3, 3),

    conv3_num_filters=64, conv3_filter_size=(3, 3),

    conv4_num_filters=64, conv4_filter_size=(3, 3),

    conv5_num_filters=64, conv5_filter_size=(3, 3),

    hidden5_num_units=100, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    hidden6_num_units=100, hidden6_nonlinearity = nonlinearities.leaky_rectify,

    output_num_units=18, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update=nesterov_momentum,
    max_epochs=20,
    on_epoch_finished = [
        EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)
    ],
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X[:, n:64-n, n:64-n, :]
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=2,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=2,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import numpy as np
from sklearn.base import BaseEstimator
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers, nonlinearities, updates, init, objectives
from nolearn.lasagne.handlers import EarlyStopping
from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1

def build_model(hyper_parameters):
    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 64, 64), # 3 = depth of input layer (color), 64x64 image
    use_label_encoder=True,
    verbose=1,
    **hyper_parameters
    )  
    return net
 
hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    hidden3_num_units=200,
    dropout3_p=0.5,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    max_epochs=200,
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)
 
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from caffezoo.googlenet import GoogleNet
from itertools import repeat
from sklearn.pipeline import make_pipeline
from scipy.ndimage.interpolation import rotate

#initialize rotation parameters

a = np.arange(64*64)
a = a.reshape(64,64)

angleToOffsetStartAndOffsetEnd = {}
alphaList = []
numRot = 8
unitRot = 360.0/numRot

for mult_alpha in range(numRot):
    alpha = mult_alpha*unitRot
    alphaList.append(alpha)

    theRot = rotate(a,alpha)
    theShape = theRot.shape

    theRest = theShape[0] - 64
    theOffsetEnd = theRest/2
    theOffsetStart = theRest - theRest/2
    
    angleToOffsetStartAndOffsetEnd[alpha] = {'indexStart':theOffsetStart,'indexEnd':theShape[0]-theOffsetEnd,'shape':theShape[0]}

def sample_from_rotation_x_old(x):
    x_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    return x_extends
 
def sample_from_rotation_y_old(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)
 
def sample_from_rotation_x(x):
    x_extends = []
    iterOnBigChunk = 0
    numBigChunk = 100
    bigChunk = x.shape[0]/numBigChunk
    for i in range(x.shape[0]):
        
        if i > iterOnBigChunk*bigChunk:
            print 'We have reached the',iterOnBigChunk,'th big chunk of',numBigChunk
            iterOnBigChunk += 1
            
        for alpha in alphaList:
            indexStart = angleToOffsetStartAndOffsetEnd[alpha]['indexStart']
            indexEnd = angleToOffsetStartAndOffsetEnd[alpha]['indexEnd']
            x_extends.append(np.array([rotate(x[i,:,:,0],alpha)[indexStart:indexEnd,indexStart:indexEnd], rotate(x[i,:,:,1],alpha)[indexStart:indexEnd,indexStart:indexEnd], rotate(x[i,:,:,2],alpha)[indexStart:indexEnd,indexStart:indexEnd]]))
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    #print x_extends.shape
    return x_extends
 
def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,numRot) )
    return np.array(y_extends)
 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb
 
def build_model():    
    L=[
        #(layers.InputLayer, {'shape':(None, 3, 64, 64)}),
        (layers.InputLayer, {'shape':(None, 3, 64, 64)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (3, 3)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':16, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (1, 1)}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 256, 'nonlinearity':nonlinearities.tanh}),
        (layers.DropoutLayer, {'p':0.2}),
        (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
    ]
 
 
    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=50,
        batch_iterator_train=FlipBatchIterator(batch_size=256),
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net
 
# currently used
def keep_dim(layers):
    #print len(layers), layers[0].shape
    return layers[0]
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = make_pipeline(
            #GoogleNet(aggregate_function=keep_dim, layer_names=["input"]),
            build_model()
        )
        
    def data_augmentation(self, X, y):
        X = sample_from_rotation_x(X)
        y = sample_from_rotation_y(y)
        return X, y
 
    def preprocess(self, X, transpose=True):
        X = (X / 255.)
        X = X.astype(np.float32)
        if transpose:
            X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)
 
    def fit(self, X, y):
        print 'Start preprocessing'
        X, y = self.preprocess(X, False), self.preprocess_y(y)
        print 'Start data augmentation'
        X, y = self.data_augmentation(X, y)
        print 'Start fit'
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
from itertools import repeat


def sample_from_rotation_x( x ):
    x_extends = []
    x_extends_extend = x_extends.extend
    rot90 = np.rot90
    np_array = np.array
    for i in range(x.shape[0]):
        x_extends.extend([
        np_array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np_array([rot90(x[i,:,:,0]),rot90(x[i,:,:,1]), rot90(x[i,:,:,2])]),
        np_array([rot90(x[i,:,:,0],2),rot90(x[i,:,:,1],2), rot90(x[i,:,:,2],2)]),
        np_array([rot90(x[i,:,:,0],3),rot90(x[i,:,:,1],3), rot90(x[i,:,:,2],3)])
        ])
    return np_array(x_extends)
 
def sample_from_rotation_y(y):
    y_extends = []
    y_extends_extend = y_extends.extend
    for i in y:
        y_extends_extend( repeat( i ,4) )
    return np.array(y_extends)


 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb
 
    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        #X = X.transpose((0, 3, 1, 2))
        X = sample_from_rotation_x( X )        
        return X
 
    def predict_preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        #X = sample_from_rotation_x( X )        
        return X

    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self
 
    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('hidden7', layers.DenseLayer),
            ('hidden8', layers.DenseLayer),
            ('hidden9', layers.DenseLayer),
            ('hidden10', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
 
hyper_parameters = dict(
    conv1_num_filters=128, conv1_filter_size=(2, 2), pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(1, 1), pool2_pool_size=(2, 2),
    conv3_num_filters=256, conv3_filter_size=(1, 1), pool3_pool_size=(2, 2),
    conv4_num_filters=128 , conv4_filter_size=(2, 2), pool4_pool_size=(4, 4),
    hidden4_num_units=1024,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=512,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    hidden6_num_units=512,
    hidden6_nonlinearity=nonlinearities.very_leaky_rectify,
    hidden6_W=init.GlorotUniform(gain='relu'),
    hidden7_num_units=256,
    hidden7_nonlinearity=nonlinearities.leaky_rectify,
    hidden7_W=init.GlorotUniform(gain='relu'),
    hidden8_num_units=128,
    hidden8_nonlinearity=nonlinearities.rectify,
    hidden8_W=init.GlorotUniform(gain='relu'),
    hidden9_num_units=64,
    hidden9_nonlinearity=nonlinearities.rectify,
    hidden9_W=init.GlorotUniform(gain='relu'),
    hidden10_num_units=32,
    hidden10_nonlinearity=nonlinearities.tanh,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        if criterion_smaller_is_better is True:
            self.best_valid = np.inf
        else:
            self.best_valid = -np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best {:s} was {:.6f} at epoch {}.".format(
                    self.criterion, self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=30,dropout1_p=0.5,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_W=init.GlorotUniform(),
    batch_iterator_train=BatchIterator(batch_size=100)

)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    dropout4_p=0.5,
    dropout5_p=0.5,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy')],
    update_momentum=0.9,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_W=init.GlorotUniform(),
    batch_iterator_train=BatchIterator(batch_size=50)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping
import skimage.color
import skimage.transform

n = 10

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),

            ('conv1', layers.Conv2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('conv5', layers.Conv2DLayer),

            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),

            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64-(2*n), 64-(2*n)),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=16, conv1_filter_size=(3, 3),

    conv2_num_filters=32, conv2_filter_size=(3, 3),

    conv3_num_filters=64, conv3_filter_size=(3, 3),

    conv4_num_filters=64, conv4_filter_size=(3, 3),

    conv5_num_filters=64, conv5_filter_size=(3, 3),

    hidden5_num_units=500, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    hidden6_num_units=500, hidden6_nonlinearity = nonlinearities.leaky_rectify,

    output_num_units=18, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update=nesterov_momentum,
    max_epochs=20,
    on_epoch_finished = [
        EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)
    ],
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X[:, n:64-n, n:64-n, :]
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        if criterion_smaller_is_better is True:
            self.best_valid = np.inf
        else:
            self.best_valid = -np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best {:s} was {:.6f} at epoch {}.".format(
                    self.criterion, self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=30,dropout1_p=0.5,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_W=init.GlorotUniform(),
    batch_iterator_train=BatchIterator(batch_size=100)

)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping
import skimage.color
import skimage.transform

n = 0

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),

            ('conv1_1', layers.Conv2DLayer),
            ('conv1_2', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),

            ('conv2_1', layers.Conv2DLayer),
            ('conv2_2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),

            ('conv3_1', layers.Conv2DLayer),
            ('conv3_2', layers.Conv2DLayer),
            ('conv3_3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),

            ('conv4_1', layers.Conv2DLayer),
            ('conv4_2', layers.Conv2DLayer),
            ('conv4_3', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),

            ('conv5_1', layers.Conv2DLayer),
            ('conv5_2', layers.Conv2DLayer),
            ('conv5_3', layers.Conv2DLayer),
            ('pool5', layers.MaxPool2DLayer),

            ('hidden6', layers.DenseLayer),
            ('hidden7', layers.DenseLayer),

            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64-(2*n), 64-(2*n)),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_1_num_filters=8, conv1_1_filter_size=(3, 3),
    conv1_2_num_filters=8, conv1_2_filter_size=(3, 3),
    pool1_pool_size=(1, 1),

    conv2_1_num_filters=16, conv2_1_filter_size=(3, 3),
    conv2_2_num_filters=16, conv2_2_filter_size=(3, 3),
    pool2_pool_size=(1, 1),

    conv3_1_num_filters=32, conv3_1_filter_size=(3, 3),
    conv3_2_num_filters=32, conv3_2_filter_size=(3, 3),
    conv3_3_num_filters=32, conv3_3_filter_size=(3, 3),
    pool3_pool_size=(1, 1),

    conv4_1_num_filters=64, conv4_1_filter_size=(3, 3),
    conv4_2_num_filters=64, conv4_2_filter_size=(3, 3),
    conv4_3_num_filters=64, conv4_3_filter_size=(3, 3),
    pool4_pool_size=(1, 1),

    conv5_1_num_filters=64, conv5_1_filter_size=(3, 3),
    conv5_2_num_filters=64, conv5_2_filter_size=(3, 3),
    conv5_3_num_filters=64, conv5_3_filter_size=(3, 3),
    pool5_pool_size=(1, 1),

    hidden6_num_units=500, hidden6_nonlinearity = nonlinearities.leaky_rectify,
    hidden7_num_units=500, hidden7_nonlinearity = nonlinearities.leaky_rectify,

    output_num_units=18, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update=nesterov_momentum,
    max_epochs=20,
    on_epoch_finished = [
        EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)
    ],
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X[:, n:64-n, n:64-n, :]
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1

lambda_regularization = 1e-5

def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden4"], layers["hidden5"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

    
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        return Xb, yb

    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.rectify,
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    output_num_units=18,
    output_W=init.Uniform((-0.01, 0.01)),
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    # objective function
    objective=objective_with_L2,
    # Optimization method:
    update=updates.adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from caffezoo.googlenet import GoogleNet
from itertools import repeat
from sklearn.pipeline import make_pipeline

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def sample_from_rotation_x(x):
    x_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    return x_extends

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb

def build_model(crop_value):    
    L=[
       (layers.InputLayer, {'shape':(None, 3, 64-2*crop_value, 64-2*crop_value)}),
       (layers.Conv2DLayer, {'num_filters':16, 'filter_size':(4,4), 'pad':0}),
       
       (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(3,3), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(3,3), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (3, 3)}),
       (layers.Conv2DLayer, {'num_filters':16, 'filter_size':(2,2), 'pad':0}),
       
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
       (layers.DropoutLayer, {'p':0.5}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
       (layers.DropoutLayer, {'p':0.5}),
       (layers.DenseLayer, {'num_units': 256, 'nonlinearity':nonlinearities.leaky_rectify}),
       (layers.DropoutLayer, {'p':0.2}),
       (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
   ] 


    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=50,
        batch_iterator_train=FlipBatchIterator(batch_size=256),
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net

# currently used
def keep_dim(layers):
    #print len(layers), layers[0].shape
    return layers[0]

class Classifier(BaseEstimator):

    def __init__(self):
        self.crop_value = 5
        self.net = make_pipeline(
            #GoogleNet(aggregate_function=keep_dim, layer_names=["input"]),
            build_model(self.crop_value)
        )
        
    def data_augmentation(self, X, y):
        X = sample_from_rotation_x(X)
        y = sample_from_rotation_y(y)
        return X, y

    def preprocess(self, X, transpose=True):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, self.crop_value:64-self.crop_value, self.crop_value:64-self.crop_value, :]
        if transpose:
            X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X, y = self.preprocess(X, False), self.preprocess_y(y)
        X, y = self.data_augmentation(X, y)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from lasagne.regularization import regularize_layer_params, l2, l1
from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)
        
                

 
lambda_regularization = 0.04
 
def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden5"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        # objective function
        objective=objective_with_L2,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=16, conv1_filter_size=(3, 3), 
    pool1_pool_size=(1, 1),
    conv2_num_filters=32, conv2_filter_size=(2, 2), 
    pool2_pool_size=(1, 1),
    conv3_num_filters=32, conv3_filter_size=(2, 2), 
    pool3_pool_size=(1, 1),
    hidden4_num_units=200, hidden4_nonlinearity = nonlinearities.leaky_rectify, 
    #hidden4_regularization = regularization.l1,
    hidden5_num_units=200, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    ##hidden5_regularization = regularization.l2,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=20,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_loss')]
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)

        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"

from sklearn.pipeline import make_pipeline
from caffezoo.googlenet import GoogleNet
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

"""
More details on Google net : http://arxiv.org/abs/1409.4842

layer_names can accept a list of strings, each string
is a layer name. The complete list of layer names
can be seen in this graph :

    https://drive.google.com/open?id=0B1CFqLHwhNoaTnVsbWtkWEhVYlE

Each node is either a convolution, a pooling layer or a nonlinearity layer, or 
other different kinds of layers.
The nodes representing convolution and pooling layers start by
the layer name (which you can put in layer_names). For convolutional layer
if you just use the name of the layer, like "conv_1" it will only take
the activations after applying the convolution. if you want to obtain
the activations after applying the activation function (ReLU), use
layername/relu, for instance conv_1/relu.

You can also provide an aggregation function, which takes
a set of layers features and returns a numpy array. The default
aggregation function used concatenate all the layers.

GoogleNet(aggregate_function=your_function, layer_names=[...])

the default aggregation function looks like this:
    def concat(layers):
        l = np.concatenate(layers, axis=1)
        return l.reshape((l.shape[0], -1))

"""

class Classifier(BaseEstimator):
 
    def __init__(self):
        self.clf = make_pipeline(
            GoogleNet(layer_names=["inception_3b/output"]),
            RandomForestClassifier(n_estimators=100, max_depth=25)
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self
 
    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb
 
    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
 
hyper_parameters = dict(
    conv1_num_filters=128, conv1_filter_size=(2, 2), pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(1, 1), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=1024,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=512,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    hidden6_num_units=256,
    hidden6_nonlinearity=nonlinearities.very_leaky_rectify,
    hidden6_W=init.GlorotUniform(gain='relu'),
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

## Simple Random Forest Classifier
class Classifier(BaseEstimator):
    def __init__(self):
        self.parameters = {
            'n_estimators' : 10,
            'max_features' : 2,
            'max_leaf_nodes' : 5
        }
        pass
 
    def fit(self, X, y):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        self.clf = RandomForestClassifier(**self.parameters)
        self.clf.fit(X_vectorized, y)
 
    def predict(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        return self.clf.predict(X_vectorized)
 
    def predict_proba(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        return self.clf.predict_proba(X_vectorized)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, objectives, updates, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
#from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
#from lasagne.regularization import regularize_layer_params, l2, l1
from nolearn.lasagne.handlers import EarlyStopping


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, ::-1, :]
        X_tmp1 = Xb[indices, :, ::-1, :]
        Y_tmp1 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, :, ::-1]
        X_tmp2 = Xb[indices, :, :, ::-1]
        Y_tmp2 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp3 = Xb[indices, :, :, :]
        Y_tmp3 = yb[indices]    
        X_tmp3 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp4 = Xb[indices, :, :, ::-1]
        Y_tmp4 = yb[indices]    
        X_tmp4 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp5 = Xb[indices, :, ::-1, :]
        Y_tmp5 = yb[indices]    
        X_tmp5 = X_tmp3.transpose((0,1,3,2))
        
        Xb = np.append(Xb,X_tmp1,axis=0)
        Xb = np.append(Xb,X_tmp2,axis=0)
        Xb = np.append(Xb,X_tmp3,axis=0)
        Xb = np.append(Xb,X_tmp4,axis=0)
        Xb = np.append(Xb,X_tmp5,axis=0)
        yb = np.append(yb,Y_tmp1)
        yb = np.append(yb,Y_tmp2)
        yb = np.append(yb,Y_tmp3)
        yb = np.append(yb,Y_tmp4)
        yb = np.append(yb,Y_tmp5)
        
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            #('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        # objective function
        # objective=objective_with_L2,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(4, 4), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(4, 4), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=1000, hidden4_nonlinearity = nonlinearities.leaky_rectify,
    #hidden4_regularization = lasagne.regularization.l2(hidden4),
    hidden5_num_units=1000, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.3,
    #hidden5_regularization = regularization.l2,
    output_num_units=18, 
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=30,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=30, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=150)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
from sklearn.base import BaseEstimator
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from lasagne import layers, nonlinearities, init, regularization
from lasagne.objectives import aggregate
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import objective
from lasagne.regularization import regularize_layer_params, l2, l1
import numpy as np

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = -np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if current_valid > self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.very_leaky_rectify,
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    update=adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)]
)

class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        X = X[:, :, 10:-10, 10:-10]
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def preprocess_train(self, X, y):
        y_train = list(self.preprocess_y(y))
        X_train = list(self.preprocess(X))
        N = len(y)
        i0 = np.random.choice(N, N/2)
        X_train.extend([X_train[i]+.1*np.random.normal(size=(44, 44)) for i in i0])
        y_train.extend([y_train[i] for i in i0])
        X_train = np.array(X_train).astype(np.float32)
        return X_train, np.array(y_train)

    def fit(self, X, y):
        X_train, y_train = self.preprocess_train(X, y)
        print('Start fitting')
        self.net.fit(X_train, y_train)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from itertools import repeat

def sample_from_rotation_x( x ):
    x_extends = []
    y_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    return np.array(x_extends)

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        if criterion_smaller_is_better is True:
            self.best_valid = np.inf
        else:
            self.best_valid = -np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best {:s} was {:.6f} at epoch {}.".format(
                    self.criterion, self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=100,dropout1_p=0.5,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_W=init.GlorotUniform(),
    batch_iterator_train=BatchIterator(batch_size=500)

)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = sample_from_rotation_x( X )
        #X = X.transpose((0, 3, 1, 2))
        return X

    def simple_preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        #X = sample_from_rotation_x( X )
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.simple_preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.simple_preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, init
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np

def build_model():
    
    L=[
        (layers.InputLayer, {'shape':(None, 3, 44, 44)}),
        (layers.Conv2DLayer, {'num_filters':64, 'filter_size':(3,3), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (3, 3)}),
        (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (4, 4)}),
        (layers.DenseLayer, {'num_units': 1024, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.7}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DenseLayer, {'num_units': 256, 'nonlinearity':nonlinearities.tanh}),
        (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
    ]
 
 
    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=100,
        batch_iterator_train=FlipBatchIterator(batch_size=256),
        on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)]
        )
    return net

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb
    
class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model()

    def preprocess(self, X):
        self.black = np.zeros([64,64,3],dtype=np.float64)
        self.white = np.ones([64,64,3],dtype=np.float64) * 255
        self.mins = np.percentile(X, 10, axis=0)
        self.maxs = np.percentile(X, 90, axis=0)
        
        X = self.filter_img(X)
        X = X[:,10:54,10:54,:]
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def filter_img(self, imgs):
        imgs = imgs.copy()
        filter_mask = self.white / (self.maxs - self.mins)
        for i in range(imgs.shape[0]):
            imgs[i] = (imgs[i]- self.mins) * filter_mask
            imgs[i] = np.maximum(imgs[i], self.black)
            imgs[i] = np.minimum(imgs[i], self.white)
            
        return imgs

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import numpy as np
from sklearn.base import BaseEstimator
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers, nonlinearities, updates, init, objectives
from nolearn.lasagne.handlers import EarlyStopping
 
def build_model(hyper_parameters):
    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 64, 64), # 3 = depth of input layer (color), 64x64 image
    use_label_encoder=True,
    verbose=1,
    **hyper_parameters
    )  
    return net
 
hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    hidden3_num_units=200,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    max_epochs=200,
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)
 
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
from itertools import repeat


def sample_from_rotation_x( x ):
    x_extends = []
    x_extends_extend = x_extends.extend
    rot90 = np.rot90
    np_array = np.array
    for i in range(x.shape[0]):
        x_extends.extend([
        np_array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np_array([rot90(x[i,:,:,0]),rot90(x[i,:,:,1]), rot90(x[i,:,:,2])]),
        np_array([rot90(x[i,:,:,0],2),rot90(x[i,:,:,1],2), rot90(x[i,:,:,2],2)]),
        np_array([rot90(x[i,:,:,0],3),rot90(x[i,:,:,1],3), rot90(x[i,:,:,2],3)])
        ])
    return np_array(x_extends)
 
def sample_from_rotation_y(y):
    y_extends = []
    y_extends_extend = y_extends.extend
    for i in y:
        y_extends_extend( repeat( i ,4) )
    return np.array(y_extends)


 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb
 
    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        #X = X.transpose((0, 3, 1, 2))
        X = sample_from_rotation_x( X )        
        return X
 
    def predict_preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        #X = sample_from_rotation_x( X )        
        return X

    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self
 
    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('hidden7', layers.DenseLayer),
            ('hidden8', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
 
hyper_parameters = dict(
    conv1_num_filters=128, conv1_filter_size=(2, 2), pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(1, 1), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=1024,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=512,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    hidden6_num_units=512,
    hidden6_nonlinearity=nonlinearities.very_leaky_rectify,
    hidden6_W=init.GlorotUniform(gain='relu'),
    hidden7_num_units=256,
    hidden7_nonlinearity=nonlinearities.leaky_rectify,
    hidden7_W=init.GlorotUniform(gain='relu'),
    hidden8_num_units=128,
    hidden8_nonlinearity=nonlinearities.rectify,
    hidden8_W=init.GlorotUniform(gain='relu'),
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    dropout4_p=0.5,
    dropout5_p=0.5,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    update_momentum=0.9,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_W=init.GlorotUniform(),
    batch_iterator_train=BatchIterator(batch_size=50)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"

from sklearn.pipeline import make_pipeline
from caffezoo.vgg import VGG
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


"""
More details on VGG : http://arxiv.org/pdf/1409.1556.pdf

layer_names can accept a list of strings, each string
is a layer name. The complete list of layer names
can be seen in this graph :

    https://drive.google.com/open?id=0B1CFqLHwhNoaODJOOEV5M1ZNbWc

Each node is either a convolution, a pooling layer or a nonlinearity layer.
The nodes representing convolution and pooling layers start by
the layer name (which you can put in layer_names). For convolutional layer
if you just use the name of the layer, like "conv_1" it will only take
the activations after applying the convolution. if you want to obtain
the activations after applying the activation function (ReLU), use
layername/relu, for instance conv_1/relu.

You can also provide an aggregation function, which takes
a set of layers features and returns a numpy array. The default
aggregation function used concatenate all the layers.

VGG(aggregate_function=your_function, layer_names=[...])

the default aggregation function looks like this:
    def concat(layers):
        l = np.concatenate(layers, axis=1)
        return l.reshape((l.shape[0], -1))

"""


class Classifier(BaseEstimator):

    def __init__(self):
        self.clf = make_pipeline(
            VGG(layer_names=["pool3"]),
            RandomForestClassifier(n_estimators=100, max_depth=25)
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self
 
    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
    
    def partial_fit(self, X, y):
        X = self.preprocess(X)
        self.net.partial_fit(X, y)
        return self

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            #('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    #dropout4_p=0.5,
    hidden5_num_units=500, 
    hidden6_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=20,
    batch_iterator_train=BatchIterator(batch_size=100),
    
    on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=2,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from caffezoo.googlenet import GoogleNet
from itertools import repeat
from sklearn.pipeline import make_pipeline

def sample_from_rotation_x(x):
    x_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    return x_extends

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb

def build_model():    
    L=[
        #(layers.InputLayer, {'shape':(None, 3, 64, 64)}),
        (layers.InputLayer, {'shape':(None, 3, 64, 64)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (3, 3)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':16, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (1, 1)}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 256, 'nonlinearity':nonlinearities.tanh}),
        (layers.DropoutLayer, {'p':0.2}),
        (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
    ]


    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=50,
        batch_iterator_train=FlipBatchIterator(batch_size=256),
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net

# currently used
def keep_dim(layers):
    #print len(layers), layers[0].shape
    return layers[0]

class Classifier(BaseEstimator):

    def __init__(self):
        self.net = make_pipeline(
            #GoogleNet(aggregate_function=keep_dim, layer_names=["input"]),
            build_model()
        )
        
    def data_augmentation(self, X, y):
        X = sample_from_rotation_x(X)
        y = sample_from_rotation_y(y)
        return X, y

    def preprocess(self, X, transpose=True):
        X = (X / 255.)
        X = X.astype(np.float32)
        if transpose:
            X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X, y = self.preprocess(X, False), self.preprocess_y(y)
        X, y = self.data_augmentation(X, y)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=15,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb

    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.very_leaky_rectify,
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
 
class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        self.clf = RandomForestClassifier(
            n_estimators=1, max_features=10, max_leaf_nodes=2) 
        self.clf.fit(X_vectorized, y)
 
    def predict(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        return self.clf.predict(X_vectorized)
 
    def predict_proba(self, X):
        X_vectorized = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))    
        return self.clf.predict_proba(X_vectorized)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb
 
    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
 
hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=1024,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=512,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    hidden6_num_units=256,
    hidden6_nonlinearity=nonlinearities.very_leaky_rectify,
    hidden6_W=init.GlorotUniform(gain='relu'),
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping
import skimage.color
import skimage.transform

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),

            ('conv1_1', layers.Conv2DLayer),
            ('conv1_2', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),

            ('conv2_1', layers.Conv2DLayer),
            ('conv2_2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),

            ('conv3_1', layers.Conv2DLayer),
            ('conv3_2', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),

            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_1_num_filters=16, conv1_1_filter_size=(3, 3),
    conv1_2_num_filters=8, conv1_2_filter_size=(3, 3),
    pool1_pool_size=(2, 2),

    conv2_1_num_filters=32, conv2_1_filter_size=(3, 3),
    conv2_2_num_filters=16, conv2_2_filter_size=(3, 3),
    pool2_pool_size=(2, 2),

    conv3_1_num_filters=64, conv3_1_filter_size=(3, 3),
    conv3_2_num_filters=32, conv3_2_filter_size=(3, 3),
    pool3_pool_size=(2, 2),

    hidden5_num_units=200, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    hidden6_num_units=200, hidden6_nonlinearity = nonlinearities.leaky_rectify,

    output_num_units=18, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=100,
    on_epoch_finished = [
        EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)
    ],
    #batch_iterator_train=TransformBatchIterator(batch_size=256)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = skimage.color.rgb2xyz(X)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=15,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, init
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from skimage.draw import circle
import scipy

def build_model():
    
    L=[
        (layers.InputLayer, {'shape':(None, 3, 54, 54)}),
        (layers.Conv2DLayer, {'num_filters':64, 'filter_size':(3,3), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (4, 4)}),
        (layers.DenseLayer, {'num_units': 500, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DenseLayer, {'num_units': 500, 'nonlinearity':nonlinearities.tanh}),
        (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
    ]
 
 
    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=100,
        batch_iterator_train=FlipBatchIterator(batch_size=100),
        on_epoch_finished=[EarlyStopping(patience=30, criterion='valid_accuracy', criterion_smaller_is_better=False)]
        )
    return net

class CropperLayer(layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input[:, :, 10:54,10:54]
    
    def get_output_shape_for(self,input_shape):
        return (input_shape[0], input_shape[1],input_shape[2]-20, input_shape[3]-20)
    
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
     
        return Xb, yb
    
class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model()
        self.circle = np.zeros((64,64), dtype=np.uint8)
        rr, cc = circle(32, 32, 27)
        self.circle[rr, cc] = 1

    def mask_circle(self, imgs):
        imgs = imgs.copy()
        for i in range(imgs.shape[0]):
            imgs[i, :, :, 0] =  imgs[i, :, :, 0] * self.circle
            imgs[i, :, :, 1] =  imgs[i, :, :, 1] * self.circle
            imgs[i, :, :, 2] =  imgs[i, :, :, 2] * self.circle
        return imgs[:, 5:59, 5:59, :]

    def preprocess(self, X):
        X = self.mask_circle(X)
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=30,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano
 
import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
from itertools import repeat


def sample_from_rotation_x( x ):
    x_extends = []
    x_extends_extend = x_extends.extend
    rot90 = np.rot90
    np_array = np.array
    for i in range(x.shape[0]):
        x_extends.extend([
        np_array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np_array([rot90(x[i,:,:,0]),rot90(x[i,:,:,1]), rot90(x[i,:,:,2])]),
        np_array([rot90(x[i,:,:,0],2),rot90(x[i,:,:,1],2), rot90(x[i,:,:,2],2)]),
        np_array([rot90(x[i,:,:,0],3),rot90(x[i,:,:,1],3), rot90(x[i,:,:,2],3)])
        ])
    return np_array(x_extends)
 
def sample_from_rotation_y(y):
    y_extends = []
    y_extends_extend = y_extends.extend
    for i in y:
        y_extends_extend( repeat( i ,4) )
    return np.array(y_extends)


 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb
 
    
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        #X = X.transpose((0, 3, 1, 2))
        X = sample_from_rotation_x( X )        
        return X
 
    def predict_preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        #X = sample_from_rotation_x( X )        
        return X

    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self
 
    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.net.predict_proba(X)
 
def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('hidden7', layers.DenseLayer),
            ('hidden8', layers.DenseLayer),
            ('hidden9', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
 
hyper_parameters = dict(
    conv1_num_filters=128, conv1_filter_size=(2, 2), pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(1, 1), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=1024,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=512,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    hidden6_num_units=512,
    hidden6_nonlinearity=nonlinearities.very_leaky_rectify,
    hidden6_W=init.GlorotUniform(gain='relu'),
    hidden7_num_units=256,
    hidden7_nonlinearity=nonlinearities.leaky_rectify,
    hidden7_W=init.GlorotUniform(gain='relu'),
    hidden8_num_units=128,
    hidden8_nonlinearity=nonlinearities.rectify,
    hidden8_W=init.GlorotUniform(gain='relu'),
    hidden9_num_units=64,
    hidden9_nonlinearity=nonlinearities.tanh,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
    
    def partial_fit(self, X, y):
        X = self.preprocess(X)
        self.net.partial_fit(X, y)
        return self

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=200, hidden5_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=2,
    
    on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from caffezoo.googlenet import GoogleNet
from itertools import repeat
from sklearn.pipeline import make_pipeline

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2).astype(np.float32)

def apply_filter(x, filt):
    return np.array([ ex * filt for ex in x])

def sample_from_rotation_x(x):
    x_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    return x_extends

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)

class FlipBatchIterator(BatchIterator):    
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 4, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb

def build_model(crop_value):    
    L=[
       (layers.InputLayer, {'shape':(None, 3, 64-2*crop_value, 64-2*crop_value)}),
       (layers.Conv2DLayer, {'num_filters':64, 'filter_size':(3,3), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (4, 4)}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify, 'W': init.GlorotUniform(gain='relu')}),
       (layers.DropoutLayer, {'p':0.5}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify, 'W': init.GlorotUniform(gain='relu')}),
       (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
   ] 

    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=100,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net

class Classifier(BaseEstimator):

    def __init__(self):
        self.crop_value = 0
        self.gaussianFilter = np.tile(makeGaussian(64-2*self.crop_value, 64-2*self.crop_value-10), (3,1,1)).transpose(1,2,0)
        self.net = build_model(self.crop_value)
        
    def data_augmentation(self, X, y):
        X = sample_from_rotation_x(X)
        y = sample_from_rotation_y(y)
        return X, y

    def preprocess(self, X, transpose=True):
        X = (X / 255.)
        #X = X[:, self.crop_value:64-self.crop_value, self.crop_value:64-self.crop_value, :]
        X = apply_filter(X, self.gaussianFilter)
        X = X.astype(np.float32)
        if transpose:
            X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X, y = self.preprocess(X, False), self.preprocess_y(y)
        X, y = self.data_augmentation(X, y)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping
import skimage.color
import skimage.transform

w = 8

class MyBatchIterator(BatchIterator):
    def transform(self, X, y):
        X = X.transpose((0, 2, 3, 1))

        X_rot, y_rot = self.rotate(X, y)
        X = np.append(X, X_rot, axis=0)
        y = np.append(y, y_rot, axis=0)

        X_flip, y_flip = self.flip(X, y)
        X = np.append(X, X_flip, axis=0)
        y = np.append(y, y_flip, axis=0)

        X_trans, y_trans = self.translate(X, y, w)

        X_crop = np.zeros((X.shape[0], 64-w, 64-w, 3))
        for i in np.arange(X.shape[0]):
            X_crop[i] = skimage.transform.resize(X[i], (64-w, 64-w))

        X = np.append(X_crop, X_trans, axis=0)
        y = np.append(y, y_trans, axis=0)

        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X, y

    def translate(self, X, y, w):
        X_trans = np.zeros((X.shape[0], 64-w, 64-w, 3))
        for i in np.arange(X.shape[0]):
            trans_x, trans_y = np.random.choice(w, 2)
            X_trans[i] = X[i, trans_x:trans_x+64-w, trans_y:trans_y+64-w, :]
        return X_trans, y

    def rotate(self, X, y):
        X_rot = np.zeros_like(X)
        for i in np.arange(X.shape[0]):
            img_rot, label_rot = self.rotateOne(X[i], y[i])
            X_rot[i] = img_rot
        return X_rot, y

    def rotateOne(self, img, label):
        angle = np.random.choice(360)
        img_rot = skimage.transform.rotate(img, angle, mode='reflect')
        return img_rot, label

    def flip(self, X, y):
        X1 = X[:, ::-1, :, :]
        X2 = X[:, :, ::-1, :]
        X3 = X[:, ::-1, ::-1, :]
        return np.concatenate((X1, X2, X3)), np.concatenate((y, y, y))


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('hidden6', layers.DenseLayer),
            ('dropout6', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64-w, 64-w),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=32, conv1_filter_size=(3, 3),
    pool1_pool_size=(2, 2),

    conv2_num_filters=64, conv2_filter_size=(3, 3),
    pool2_pool_size=(2, 2),

    conv3_num_filters=128, conv3_filter_size=(3, 3),
    pool3_pool_size=(1, 1),

    hidden5_num_units=200,
    hidden6_num_units=200,

    output_num_units=18, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update=nesterov_momentum,
    max_epochs=30,
    on_epoch_finished = [
        EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)
    ],
    batch_iterator_train=MyBatchIterator(batch_size=100),
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import numpy as np
from sklearn.base import BaseEstimator
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers, nonlinearities, updates, init, objectives
from nolearn.lasagne.handlers import EarlyStopping
 
 
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb

 
def build_model(hyper_parameters):
    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 64, 64), # 3 = depth of input layer (color), 64x64 image
    use_label_encoder=True,
    verbose=1,
    **hyper_parameters
    )  
    return net
 
hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    hidden3_num_units=200,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    max_epochs=200,
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=256),
)
 
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np

''' class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)
'''


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
#            ('pool1', layers.MaxPool2DLayer),
#            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
#            ('pool3', layers.MaxPool2DLayer),
#            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('conv5', layers.Conv2DLayer),
            ('pool5', layers.MaxPool2DLayer),
            ('hidden6', layers.DenseLayer),
            ('dropout6', layers.DropoutLayer),
            ('hidden7', layers.DenseLayer),
            ('dropout7', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44), # 64 to 44 because cropping input images with 10:54 x 10:54
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=32, conv1_filter_size=(5, 5),
#    pool1_pool_size=(1, 1),
#    conv2_num_filters=32, conv2_filter_size=(5, 5),
    pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(3, 3),
#    pool3_pool_size=(1, 1),
#    conv4_num_filters=64, conv4_filter_size=(3, 3),
    pool4_pool_size=(2, 2),
    conv5_num_filters=128, conv5_filter_size=(2, 2),
    pool5_pool_size=(2, 2),
    hidden6_num_units=300, dropout6_p=0.5, hidden6_nonlinearity = nonlinearities.leaky_rectify,
    #hidden6_regularization = regularization.l1,
    hidden7_num_units=300, dropout7_p=0.5, hidden7_nonlinearity = nonlinearities.leaky_rectify,
    #hidden7_regularization = regularization.l2,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=30,

    # handlers
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', criterion_smaller_is_better=False)]
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
    
    def partial_fit(self, X, y):
        X = self.preprocess(X)
        self.net.partial_fit(X, y)
        return self

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=30,
    
    on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, objectives, updates, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
#from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
#from lasagne.regularization import regularize_layer_params, l2, l1
from nolearn.lasagne.handlers import EarlyStopping
from skimage import data
from skimage import transform

lambda_regularization = 0.01

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, ::-1, :]
        X_tmp1 = Xb[indices, :, ::-1, :]
        Y_tmp1 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, :, ::-1]
        X_tmp2 = Xb[indices, :, :, ::-1]
        Y_tmp2 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp3 = Xb[indices, :, :, :]
        Y_tmp3 = yb[indices]    
        X_tmp3 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp4 = Xb[indices, :, :, ::-1]
        Y_tmp4 = yb[indices]    
        X_tmp4 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp5 = Xb[indices, :, ::-1, :]
        Y_tmp5 = yb[indices]    
        X_tmp5 = X_tmp3.transpose((0,1,3,2))
        
        Xb = np.append(Xb,X_tmp1,axis=0)
        Xb = np.append(Xb,X_tmp2,axis=0)
        Xb = np.append(Xb,X_tmp3,axis=0)
        Xb = np.append(Xb,X_tmp4,axis=0)
        Xb = np.append(Xb,X_tmp5,axis=0)
        yb = np.append(yb,Y_tmp1)
        yb = np.append(yb,Y_tmp2)
        yb = np.append(yb,Y_tmp3)
        yb = np.append(yb,Y_tmp4)
        yb = np.append(yb,Y_tmp5)
        
        # small rotation of the images
        lx = 44
        pad_lx = 64
        shift_x = lx/2.
        shift_y = lx/2.
        
        
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp6 = Xb[indices, :, ::-1, :]
        X_tmp6 = X_tmp6.transpose(0,2,3,1)
        X_tmp6 = np.pad(X_tmp6,((0,0),(10,10),(10,10),(0,0)),'constant', constant_values=(0,0))
        Y_tmp6 = yb[indices]
        x_rot = X_tmp6[0]
        x_rot = x_rot.reshape(1,pad_lx,pad_lx,3)
        
        
        # tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(15))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        
        for i in X_tmp6[1::]:
            tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(np.random.randint(30)-15))
            xdel = transform.warp(i, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)            
            xdel=xdel.reshape(1,pad_lx,pad_lx,3)
            x_rot=np.append(x_rot,xdel,axis=0)
        
        x_rot = x_rot[:, 10:54, 10:54, :]
        x_rot = x_rot.transpose(0,3,1,2)
        x_rot = x_rot.astype(np.float32)
        Xb = np.append(Xb,x_rot,axis=0)
        yb = np.append(yb,Y_tmp6)
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            #('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        # objective function
        # objective=objective_with_L2,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(4, 4), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(4, 4), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=1000, hidden4_nonlinearity = nonlinearities.leaky_rectify,
    #hidden4_regularization = lasagne.regularization.l2(hidden4),
    hidden5_num_units=1000, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.3,
    #hidden5_regularization = regularization.l2,
    output_num_units=18, 
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=100,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=40, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=150)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        
        
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

import math
from skimage import data
from skimage import transform as tf

from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1
 
lambda_regularization = 0.04
 
def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden5"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        return Xb, yb


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        # additional static transformations
        print 'start preprocess'
        tformslist = list()
        tformslist.append(tf.SimilarityTransform(scale=1))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = math.pi/10))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = -math.pi/10))

        X_new = np.zeros((X.shape[0] * 3, X.shape[1], X.shape[2], X.shape[3]))
        print 'X shape ', X.shape[0]
        for i in xrange(X.shape[0]):
            Xbase = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
            Xbase[10:54,10:54,:] = X[i,10:54,10:54,:]
            if i % 1000 == 0:
                print 'performed first ' + str((i)) + ' transformations.'
            for j in xrange(len(tformslist)):
                X_new[len(tformslist)*i + j, :, :, :] = tf.warp(Xbase, tformslist[j])
        print 'end preprocess'
        X_new = (X_new[:,10:54,10:54,:] / 255.)
        X_new = X_new.astype(np.float32)
        X_new = X_new.transpose((0, 3, 1, 2))
        return X_new
    
    def preprocess_test(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        y_new = np.zeros((y.shape[0] * 3))
        for i in xrange(y.shape[0]):
            for j in xrange(3):
                y_new[3*i + j] = y[i]
        return y_new.astype(np.int32)

    def fit(self, X, y):
        X_new = self.preprocess(X)
        self.net.fit(X_new, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess_test(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess_test(X)
        return self.net.predict_proba(X)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        objective=objective_with_L2,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.very_leaky_rectify,
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.leaky_rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),

            ('conv1_1', layers.Conv2DLayer),
            ('conv1_2', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),

            ('conv2_1', layers.Conv2DLayer),
            ('conv2_2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),

            ('conv3_1', layers.Conv2DLayer),
            ('conv3_2', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),

            ('hidden5', layers.DenseLayer),
            ('hidden6', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_1_num_filters=16, conv1_1_filter_size=(3, 3),
    conv1_2_num_filters=8, conv1_2_filter_size=(3, 3),
    pool1_pool_size=(2, 2),

    conv2_1_num_filters=32, conv2_1_filter_size=(3, 3),
    conv2_2_num_filters=16, conv2_2_filter_size=(3, 3),
    pool2_pool_size=(2, 2),

    conv3_1_num_filters=64, conv3_1_filter_size=(3, 3),
    conv3_2_num_filters=32, conv3_2_filter_size=(3, 3),
    pool3_pool_size=(2, 2),

    hidden5_num_units=500,
    hidden6_num_units=500,

    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=30,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, objectives, updates, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
#from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
#from lasagne.regularization import regularize_layer_params, l2, l1
from nolearn.lasagne.handlers import EarlyStopping
from skimage import data
from skimage import transform

lambda_regularization = 0.01

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, ::-1, :]
        X_tmp1 = Xb[indices, :, ::-1, :]
        Y_tmp1 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, :, ::-1]
        X_tmp2 = Xb[indices, :, :, ::-1]
        Y_tmp2 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp3 = Xb[indices, :, :, :]
        Y_tmp3 = yb[indices]    
        X_tmp3 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp4 = Xb[indices, :, :, ::-1]
        Y_tmp4 = yb[indices]    
        X_tmp4 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp5 = Xb[indices, :, ::-1, :]
        Y_tmp5 = yb[indices]    
        X_tmp5 = X_tmp3.transpose((0,1,3,2))
        
        Xb = np.append(Xb,X_tmp1,axis=0)
        Xb = np.append(Xb,X_tmp2,axis=0)
        Xb = np.append(Xb,X_tmp3,axis=0)
        Xb = np.append(Xb,X_tmp4,axis=0)
        Xb = np.append(Xb,X_tmp5,axis=0)
        yb = np.append(yb,Y_tmp1)
        yb = np.append(yb,Y_tmp2)
        yb = np.append(yb,Y_tmp3)
        yb = np.append(yb,Y_tmp4)
        yb = np.append(yb,Y_tmp5)
        
        # small rotation of the images
        lx = 44
        pad_lx = 64
        shift_x = lx/2.
        shift_y = lx/2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(15))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp6 = Xb[indices, :, ::-1, :]
        X_tmp6 = X_tmp6.transpose(0,2,3,1)
        X_tmp6 = np.pad(X_tmp6,((0,0),(10,10),(10,10),(0,0)),'constant', constant_values=(0,0))
        Y_tmp6 = yb[indices]
        x_rot = X_tmp6[0]
        x_rot = x_rot.reshape(1,pad_lx,pad_lx,3)
        for i in X_tmp6[1::]:
            xdel = transform.warp(i, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)            
            xdel=xdel.reshape(1,pad_lx,pad_lx,3)
            x_rot=np.append(x_rot,xdel,axis=0)
        
        x_rot = x_rot[:, 10:54, 10:54, :]
        x_rot = x_rot.transpose(0,3,1,2)
        x_rot = x_rot.astype(np.float32)
        Xb = np.append(Xb,x_rot,axis=0)
        yb = np.append(yb,Y_tmp6)
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            #('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        # objective function
        # objective=objective_with_L2,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(4, 4), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(4, 4), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=1000, hidden4_nonlinearity = nonlinearities.leaky_rectify,
    #hidden4_regularization = lasagne.regularization.l2(hidden4),
    hidden5_num_units=1000, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.3,
    #hidden5_regularization = regularization.l2,
    output_num_units=18, 
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=150,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=40, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=150)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        
        
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
    
    def partial_fit(self, X, y):
        X = self.preprocess(X)
        self.net.partial_fit(X, y)
        return self

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=200, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=2,
    
    on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)


import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet
import numpy as np




class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond is True:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)

def build_model():
    hyper_parameters = dict(
    #hidden4_regularization = regularization.l1,
    #hidden5_regularization = regularization.l2,
    
    # handlers
)
    
    L=[
        (layers.InputLayer, {'shape':(None, 3, 64, 64)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (3, 3)}),
        (layers.Conv2DLayer, {'num_filters':32, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
        (layers.Conv2DLayer, {'num_filters':16, 'filter_size':(2,2), 'pad':0}),
        (layers.MaxPool2DLayer, {'pool_size': (1, 1)}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify}),
        (layers.DropoutLayer, {'p':0.5}),
        (layers.DenseLayer, {'num_units': 256, 'nonlinearity':nonlinearities.tanh}),
        (layers.DropoutLayer, {'p':0.2}),
        (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
    ]


    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=30,
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net




class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model()

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from caffezoo.googlenet import GoogleNet
from itertools import repeat
from sklearn.pipeline import make_pipeline

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2).astype(np.float32)

def apply_filter(x, filt):
    return np.array([ ex * filt for ex in x])

def sample_from_rotation_x(x):
    x_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    return x_extends

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)

class FlipBatchIterator(BatchIterator):    
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 4, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb

def build_model(crop_value):    
    L=[
       (layers.InputLayer, {'shape':(None, 3, 64-2*crop_value, 64-2*crop_value)}),
       (layers.Conv2DLayer, {'num_filters':64, 'filter_size':(3,3), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (4, 4)}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify, 'W': init.GlorotUniform(gain='relu')}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify, 'W': init.GlorotUniform(gain='relu')}),
       (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
   ] 

    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=100,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net

class Classifier(BaseEstimator):

    def __init__(self):
        self.crop_value = 0
        self.gaussianFilter = np.tile(makeGaussian(64-2*self.crop_value, 64-2*self.crop_value-10), (3,1,1)).transpose(1,2,0)
        self.net = build_model(self.crop_value)
        
    def data_augmentation(self, X, y):
        X = sample_from_rotation_x(X)
        y = sample_from_rotation_y(y)
        return X, y

    def preprocess(self, X, transpose=True):
        X = (X / 255.)
        #X = X[:, self.crop_value:64-self.crop_value, self.crop_value:64-self.crop_value, :]
        X = apply_filter(X, self.gaussianFilter)
        X = X.astype(np.float32)
        if transpose:
            X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X, y = self.preprocess(X, False), self.preprocess_y(y)
        X, y = self.data_augmentation(X, y)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)

import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping

import math
from skimage import data
from skimage import transform as tf

from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1
 
lambda_regularization = 0.04
 
def objective_with_L2(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden5"]], l2)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    
    if deterministic is False:
        return loss + reg * lambda_regularization
    else:
        return loss

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        return Xb, yb


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        # additional static transformations
        print 'start preprocess'
        tformslist = list()
        tformslist.append(tf.SimilarityTransform(scale=1))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = math.pi/10))
        tformslist.append(tf.SimilarityTransform(scale=1, rotation = -math.pi/10))

        X_new = np.zeros((X.shape[0] * 3, X.shape[1], X.shape[2], X.shape[3]))
        print 'X shape ', X.shape[0]
        for i in xrange(X.shape[0]):
            Xbase = np.zeros((X.shape[1], X.shape[2], X.shape[3]))
            Xbase[10:54,10:54,:] = X[i,10:54,10:54,:]
            if i % 1000 == 0:
                print 'performed first ' + str((i)) + ' transformations.'
            for j in xrange(len(tformslist)):
                X_new[len(tformslist)*i + j, :, :, :] = tf.warp(Xbase, tformslist[j])
        print 'end preprocess'
        X_new = (X_new[:,10:54,10:54,:] / 255.)
        X_new = X_new.astype(np.float32)
        X_new = X_new.transpose((0, 3, 1, 2))
        return X_new
    
    def preprocess_test(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        y_new = np.zeros((y.shape[0] * 3))
        for i in xrange(y.shape[0]):
            for j in xrange(3):
                y_new[3*i + j] = y[i]
        return y_new.astype(np.int32)

    def fit(self, X, y):
        X_new = self.preprocess(X)
        self.net.fit(X_new, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess_test(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess_test(X)
        return self.net.predict_proba(X)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        objective=objective_with_L2,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.very_leaky_rectify,
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.leaky_rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    hidden5_nonlinearity=nonlinearities.leaky_rectify,
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=100)
)
import os
import theano
import theano.tensor as T
from collections import OrderedDict
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, regularization
from lasagne.objectives import aggregate
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import objective
from lasagne.regularization import regularize_layer_params, l2, l1
import numpy as np

lambda_regularization = 1e-3


def get_or_compute_grads(loss_or_grads, params):
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def apply_prox(loss_or_grads, params, learning_rate, momentum=.5, reg_l1=.1):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable, borrow=True)
        x = momentum * velocity - learning_rate * grad
        updates[velocity] = x
        updates[param] = T.sgn(param+x)*T.maximum(abs(param+x)-reg_l1, 0)

    return updates


def objective_with_L1(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden4"], layers["hidden5"]], l1)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    return loss + reg * lambda_regularization


class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 54, 54),
        use_label_encoder=True,
        verbose=1,
        #objective=objective_with_L1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    # Conv layer 1
    conv1_num_filters=64,
    conv1_filter_size=(3, 3),
    pool1_pool_size=(2, 2),

    # Conv layer 2
    conv2_num_filters=128,
    conv2_filter_size=(2, 2),
    pool2_pool_size=(2, 2),

    # Conv layer 3
    conv3_num_filters=128,
    conv3_filter_size=(2, 2),
    pool3_pool_size=(4, 4),

    # Layer 4
    hidden4_num_units=500,
    hidden4_nonlinearity = nonlinearities.leaky_rectify,
    dropout4_p=0,

    # Layer 5
    hidden5_num_units=500,
    hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.5,

    # Output
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,

    # Training
    #update= apply_prox,
    update_learning_rate=0.1,
    update_momentum=0.9,
    #update_reg_l1=.001,
    max_epochs=150,

    # handlers
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_loss')]
)


class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        X = X[:, :, 5:-5, 5:-5]
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        print('Start fitting')
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

import numpy as np
from matplotlib.colors import rgb_to_hsv
from sklearn.base import BaseEstimator
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import colorsys

# pixels to crop in the 4 borders
n = 12

class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X[:, n:64-n, n:64-n, :]
        X = X ** (2)
        
        X = X.transpose((0, 3, 1, 2))        
        return X

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64-2*n, 64-2*n),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update=updates.adagrad,
    max_epochs=30,
    
    on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)

