import numpy as np

import theano
import theano.tensor as T

from lasagne import layers, nonlinearities, init
from lasagne.layers import batch_norm

from helpers import floatX, softmax

rectify = nonlinearities.rectify
sigmoid = nonlinearities.sigmoid
linear = lambda x:x
very_leaky_rectify = nonlinearities.very_leaky_rectify

def build_mask(shape, type='a', n_channels=1):
    # Inspired by : https://github.com/igul222/pixel_rnn/blob/master/pixel_rnn.py

    h = shape[2]
    w = shape[3]
    m = np.zeros(shape)
    m[:, :, 0:h/2, :] = 1
    m[:, :, h/2, 0:w/2 + 1] = 1
    if type == 'a':
        # disable center of the filter if mask type is 'a'
        m[:, :, h/2, w/2] = 0
        
        # so far all centers are zero, make
        # one the ones which relate a next color
        # with a previous color
        
        #red(input)->green(output)
        #m[1::n_channels, 0::n_channels, h/2, w/2] = 1
        #green(input)->blue(output)
        #m[2::n_channels, 1::n_channels, h/2, w/2] = 1
        #red(input)->blue(output)
        #m[2::n_channels, 0::n_channels, h/2, w/2] = 1
        for i in range(n_channels):
            for j in range(n_channels):
                if i > j:
                    m[i::n_channels, j::n_channels, h/2, w/2] = 1.
    elif type == 'b':
        # so far all centers are one, make
        # zero all the forbidden ones

        #green(input)->red(output)
        #m[0::n_channels, 1::n_channels, h/2, w/2] = 0
        #blue(input)->red(output)
        #m[0::n_channels, 2::n_channels, h/2, w/2] = 0
        #blue(input)->green(output)
        #m[1::n_channels, 2::n_channels, h/2, w/2] = 0
        for i in range(n_channels):
            for j in range(n_channels):
                if i < j:
                    m[i::n_channels, j::n_channels, h/2, w/2] = 0.
    return floatX(m)

def build_mask2(shape, type='a', n_channels=1):
    mask_type = type
    output_dim, input_dim, filter_size, _ = shape 
    mask = np.ones(
        (output_dim, input_dim, filter_size, filter_size), 
        dtype=theano.config.floatX
    )
    center = filter_size//2
    for i in xrange(filter_size):
        for j in xrange(filter_size):
                if (j > center) or (j==center and i > center):
                    mask[:, :, j, i] = 0.
    for i in xrange(n_channels):
        for j in xrange(n_channels):
            if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                mask[
                    j::n_channels,
                    i::n_channels,
                    center,
                    center
                ] = 0.
    return floatX(mask)

conv2d = layers.Conv2DLayer
local2d = layers.local.LocallyConnected2DLayer

def masked_conv2d(layer, 
                  num_filters, 
                  filter_size, 
                  W=init.GlorotUniform(gain='relu'), 
                  type='a',
                  n_channels=1,
                  **kw):
    shape = (num_filters, layer.output_shape[1]) + filter_size
    if hasattr(W, '__call__'):
        W = W(shape)
    mask = build_mask2((num_filters, shape[1]) + filter_size, type=type, n_channels=n_channels)
    W = mask * theano.shared(W)
    layer = layers.Conv2DLayer(
        layer,
        num_filters=num_filters,
        filter_size=filter_size,
        W=W,
        flip_filters=False,
        **kw)
    return layer

def build_model_pixelcnn(input_shape=(None, 1, 32, 32), nb_layers=4, nb_outputs=1, dim=64, mask_channels=True):
    nc = input_shape[1] if mask_channels else 1 # nb of channels
    inp = layers.InputLayer(input_shape)
    conv = inp
    conv = masked_conv2d(conv, dim * nc, (7, 7), type='a', n_channels=nc, nonlinearity=rectify, pad='same')
    for _ in range(nb_layers):
        conv = masked_conv2d(conv, dim * nc, (3, 3), type='b', n_channels=nc, nonlinearity=rectify, pad='same')
    conv = masked_conv2d(conv, dim * nc, (1, 1), type='b', n_channels=nc, nonlinearity=rectify, pad='same')
    conv = masked_conv2d(conv, dim * nc, (1, 1), type='b', n_channels=nc, nonlinearity=rectify, pad='same')
    conv = masked_conv2d(conv, nb_outputs * nc, (1, 1), type='b', n_channels=nc, W=init.GlorotUniform(), nonlinearity=linear, pad='same')
    out = layers.ReshapeLayer(conv, ([0], nb_outputs, nc, [2], [3]))
    return inp, out

def equals_(x, y, eps=1e-8):
    return T.abs_(x - y) <= eps

def wta_spatial(X):
    # From http://arxiv.org/pdf/1409.2752v2.pdf
    # Introduce sparsity for each channel/feature map, for each
    # feature map make all activations zero except the max
    # This introduces a sparsity level dependent on the number
    # of feature maps,
    # for instance if we have 10 feature maps, the sparsity level
    # is 1/10 = 10%
    mask = (equals_(X, T.max(X, axis=(2, 3), keepdims=True))) * 1
    return X * mask

def wta_channel_strided(stride=2):

    def apply_(X):
        B, F = X.shape[0:2]
        w, h = X.shape[2:]
        X_ = X.reshape((B, F, w / stride, stride, h / stride, stride))
        mask = equals_(X_, X_.max(axis=(1, 3, 5), keepdims=True)) * 1
        mask = mask.reshape(X.shape)
        return X * mask
    return apply_

def build_model_aa(input_shape=(None, 1, 32, 32), nb_layers=4, nb_outputs=1, dim=64, **kw):
    nc = 1
    inp = layers.InputLayer(input_shape)
    conv = inp
    for _ in range(nb_layers):
        conv = conv2d(conv, dim * nc, (5, 5), nonlinearity=rectify)
    conv = layers.NonlinearityLayer(conv, wta_spatial)
    for i in range(nb_layers):
        conv = conv2d(conv, dim * nc, (5, 5), nonlinearity=rectify, pad='full')
    conv = conv2d(conv, nb_outputs * nc, (1, 1), W=init.GlorotUniform(), nonlinearity=linear)
    out = layers.ReshapeLayer(conv, ([0], nb_outputs, nc, [2], [3]))
    return inp, out

def build_model_aa_vertebrate(input_shape=(None, 1, 32, 32), nb_layers=4, nb_outputs=1, dim=64, **kw):
    inp = layers.InputLayer(input_shape)
    conv = inp
    outs = []
    for i in range(nb_layers):
        conv = conv2d(conv, dim, (5, 5), nonlinearity=rectify, pad='same')
        if i == nb_layers - 1:
            conv_sparse = layers.NonlinearityLayer(conv, linear)
            r = 4 * (i + 1)
            conv_out = conv2d(conv_sparse, nb_outputs, (r + 1, r +  1), W=init.GlorotUniform(), nonlinearity=linear, pad='same')
            #conv_out = layers.NonlinearityLayer(conv_out, softmax)
            outs.append(conv_out)
            print(conv_out.output_shape)
    out = layers.ElemwiseSumLayer(outs)
    out = layers.ReshapeLayer(out, ([0], nb_outputs, 1, [2], [3]))
    return inp, out

def build_model_aa_vertebrate_2(input_shape=(None, 1, 32, 32), nb_layers=4, nb_outputs=1, dim=64, **kw):
    inp = layers.InputLayer(input_shape)
    conv = inp
    outs = []
    for i in range(nb_layers):
        conv = conv2d(conv, dim, (5, 5), nonlinearity=rectify, pad='same')
        if i == nb_layers - 1:
            conv_sparse = layers.NonlinearityLayer(conv, linear)
            r = 4 * (i + 1)
            conv_out = conv2d(conv_sparse, nb_outputs, (r + 1, r +  1), W=W, nonlinearity=linear, pad='same')
            outs.append(conv_out)
            print(conv_out.output_shape)
    out = layers.ElemwiseSumLayer(outs)
    out = layers.ReshapeLayer(out, ([0], nb_outputs, 1, [2], [3]))
    return inp, out


if __name__ == '__main__':
    import theano
    import theano.tensor as T
    inp, out = build_model_aa(input_shape=(None, 3, 32, 32), nb_layers=4, nb_outputs=256)
    X = T.tensor4()
    y = layers.get_output(out, X)
    fn = theano.function([X], y)
    x = np.random.uniform(size=(1, 3, 32,32))
    x = floatX(x)
    y = fn(x)
    print(y.shape)
