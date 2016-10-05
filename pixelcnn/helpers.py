import numpy as np
from skimage.io import imsave
from skimage.util import pad

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
    out = np.empty(len(x))
    out = intX(out)
    for i in range(len(x)):
        p = x[i]
        out[i] = rng.choice(np.arange(len(p)), p=p)
    return out

class ColorDiscretizerJoint(object):
    
    def __init__(self, centers, batch_size=1000):
        # assume centers has shape (nb_centers, nb_channels)
        self.centers = np.array(centers)
        self.batch_size = batch_size
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assume X has shape (nb_examples, nb_channels, h, w)
        X = X[:, :, :, :, np.newaxis] #(nb_examples, nb_channels, h, w, 1)
        centers = self.centers.T # (nb_channels, nb_centers)
        centers = centers[np.newaxis, :, np.newaxis, np.newaxis, :]#(1, nb_channels, 1, 1, nb_centers)
        outputs = []
        for i in range(0, len(X), self.batch_size):
            dist = np.abs(X[i:i + self.batch_size] - centers) # (nb_examples, nb_channels, h, w, nb_centers)
            dist = dist.sum(axis=1) # (nb_examples, h, w, nb_centers)
            out = dist.argmin(axis=3) # (nb_examples, h, w)
            outputs.append(out)
        return np.concatenate(outputs, axis=0)

    def inverse_transform(self, X):
        # assume X has shape (nb_examples, h, w)
        X = intX(X)
        nb, h, w = X.shape
        X = X.flatten()
        X = self.centers[X]
        nb_channels = X.shape[1]
        X = X.reshape((nb, h, w, nb_channels))
        X = X.transpose((0, 3, 1, 2))
        return X # (nb_examples, nb_channels, h, w)

class ColorDiscretizerPerChannel(object):
    
    def __init__(self, centers, batch_size=1000):
        # assume centers has shape (nb_centers, nb_channels)
        self.centers = np.array(centers)
        self.batch_size = batch_size
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assume X has shape (nb_examples, nb_channels, h, w)
        out = np.empty_like(X)
        nb_channels = X.shape[1]
        for channel in range(nb_channels):
            X = X[:, channel, :, :, np.newaxis] #(nb_examples, h, w, 1)
            centers = self.centers[:, channel] # (nb_centers,)
            centers = centers[np.newaxis, np.newaxis, np.newaxis, :]#(1, 1, 1, nb_centers)
            outputs = []
            for i in range(0, len(X), self.batch_size):
                dist = np.abs(X[i:i + self.batch_size] - centers) # (nb_examples, h, w, nb_centers)
                out[i:i + self.batch_size, channel, :, :] = dist.argmin(axis=3) # (nb_examples, h, w)
        return out

    def inverse_transform(self, X):
        # assume X has shape (nb_examples, nb_channels, h, w)
        X = intX(X)
        nb_examples, nb_channels, h, w = X.shape
        out = np.empty_like(X)
        for channel in range(nb_channels):
            x = X[:, channel].flatten()
            x = self.centers[:, channel][x]
            x = x.reshape((nb_examples, h, w))
            out[:, channel, :, :] = x
        return out

class ColorDiscretizerRound(object):

    def __init__(self):
        pass
    
    def fit(self, X):
        return self

    def transform(self, X):
        return intX(X)

    def inverse_transform(self, X):
        return intX(X)

def color_discretization(X, n_bins, method='kmeans'):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    kmeans = MiniBatchKMeans
    # assume X has shape (nb_examples, nb_colors, h, w)
    X = X.transpose((0, 2, 3, 1))
    nb, h, w, nb_colors = X.shape
    X = X.reshape((nb * h * w, nb_colors))
    clus = kmeans(n_clusters=n_bins).fit(X)
    return clus.cluster_centers_ # (n_bins, nb_colors)

def categ(X, D=10):
    X = intX(X)
    nb = np.prod(X.shape)
    x = X.flatten()
    m = np.zeros((nb, D))
    m[np.arange(nb), x] = 1.
    m = m.reshape(X.shape + (D,))
    m = floatX(m)
    return m

def softmax(x, axis=1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def floatX(x):
    return np.array(x, dtype=theano.config.floatX)

def intX(x):
    return np.array(x, dtype='int32')

def random_crop(X, shape=(8, 8), rng=np.random):
    # assumes x is (h, w, nb_channels)
    py, px = shape[0] / 2, shape[1] / 2
    X_ = np.zeros((X.shape[0] + py * 2, X.shape[1] + px * 2, X.shape[2]))
    x = rng.randint(py, X.shape[1])
    y = rng.randint(px, X.shape[0])
    X_[py:-py, px:-px, :] = X
    X_ = X_[y - py:y + py, x- px:x + px]
    return X_
