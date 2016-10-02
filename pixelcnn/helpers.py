import numpy as np
from skimage.io import imsave

import theano
import theano.tensor as T

from lasagne import objectives
from lasagne import layers
from lasagne.updates import adam
from lasagne import nonlinearities

def disp(x, **kw):
    # x shape : (examples, color, h, w)
    from lasagnekit.misc.plot_weights import dispims_color
    x = x.transpose((0, 2, 3, 1))
    x = x * np.ones((1, 1, 1, 3))
    x = dispims_color(x,  **kw)
    return x

def sample_multinomial(x, rng=np.random):
    out = np.empty(len(x), dtype='int32')
    for i in range(len(x)):
        p = x[i]
        out[i] = rng.choice(np.arange(len(p)), p=p)
    return out

def generate(X, predict_fn, sample_fn=sample_multinomial):
    X = floatX(X)
    out = np.empty((X.shape[0],) + (X.shape[2:]), dtype='int32')
    nb, nb_channels, h, w = X.shape
    for y in range(h):
        for x in range(w):
            p = predict_fn(X)
            sample = sample_fn(p[:, :, y, x])
            X[:, :, y, x] = categ(sample, D=nb_channels)
            out[:, y, x] = sample
    return out

class ColorDiscretizer(object):
    
    def __init__(self, centers):
        # assume centers has shape (nb_centers, nb_channels)
        self.centers = np.array(centers)
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assume X has shape (nb_examples, nb_channels, h, w)
        X = X[:, :, :, :, np.newaxis] #(nb_examples, nb_channels, h, w, 1)
        centers = self.centers.T # (nb_channels, nb_centers)
        centers = centers[np.newaxis, :, np.newaxis, np.newaxis, :]#(1, nb_channels, 1, 1, nb_centers)
        dist = np.abs(X - centers) # (nb_examples, nb_channels, h, w, nb_centers)
        dist = dist.sum(axis=1) # (nb_examples, h, w, nb_centers)
        out = dist.argmin(axis=3) # (nb_examples, h, w)
        return out

    def inverse_transform(self, X):
        # assume X has shape (nb_examples, h, w)
        nb, h, w = X.shape
        X = X.flatten()
        X = self.centers[X]
        nb_channels = X.shape[1]
        X = X.reshape((nb, h, w, nb_channels))
        X = X.transpose((0, 3, 1, 2))
        return X
        
def color_discretization(X, n_bins):
    from sklearn.cluster import KMeans
    # assume X has shape (nb_examples, nb_colors, h, w)
    X = X.transpose((0, 2, 3, 1))
    nb, h, w, nb_colors = X.shape
    X = X.reshape((nb * h * w, nb_colors))
    clus = KMeans(n_clusters=n_bins).fit(X)
    return clus.cluster_centers_ # (n_bins, nb_colors)

def categ(X, D=10):
    nb = np.prod(X.shape)
    x = X.flatten()
    m = np.zeros((nb, D))
    m[np.arange(nb), x] = 1.
    m = m.reshape(X.shape + (D,))
    m = floatX(m)
    return m

def softmax(x):
    # x has shape (nb_examples, nb_channels, h, w)
    nb, nb_channels, h, w = x.shape
    x = x.transpose((0, 2, 3, 1))
    x = x.reshape((nb * h * w, nb_channels))
    x = T.nnet.softmax(x)
    x = x.reshape((nb, h, w, nb_channels))
    x = x.transpose((0, 3, 1, 2))
    return x

def floatX(x):
    return np.array(x, dtype=theano.config.floatX)
