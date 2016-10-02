import numpy as np

import theano

from lasagne import layers, nonlinearities, init

from helpers import floatX

rectify = nonlinearities.rectify
sigmoid = nonlinearities.sigmoid
linear = lambda x:x

def build_mask(shape, type='a'):
    h = shape[2]
    w = shape[3]
    m = np.zeros(shape)
    m[:, :, 0:h/2, :] = 1
    m[:, :, h/2, 0:w/2 + 1] = 1
    # disable center of the filter if mask type is 'a'
    if type == 'a': m[:, :, h/2, w/2] = 0
    return floatX(m)

def masked_conv2d(layer, 
                  num_filters, 
                  filter_size, 
                  nonlinearity=nonlinearities.rectify, 
                  W=init.GlorotUniform(gain='relu'), 
                  type='a',
                  **kw):
    shape = (num_filters, layer.output_shape[1]) + filter_size
    if hasattr(W, '__call__'):
        W = W(shape)
    mask = build_mask((num_filters, shape[1]) + filter_size, type=type)
    W = mask * theano.shared(W)
    layer = layers.Conv2DLayer(
        layer,
        num_filters=num_filters,
        filter_size=filter_size,
        nonlinearity=nonlinearity,
        W=W,
        flip_filters=False,
        **kw)
    return layer

def build_model_pixelcnn(input_shape=(None, 1, 32, 32), n_outputs=1):
    inp = layers.InputLayer(input_shape)
    conv = inp
    conv = masked_conv2d(conv, 64, (7, 7), type='a', nonlinearity=rectify, pad='same')
    for _ in range(4):
        conv = masked_conv2d(conv, 64, (3, 3), type='b', nonlinearity=rectify, pad='same')
    conv = masked_conv2d(conv, 64, (1, 1), type='b', nonlinearity=rectify, pad='same')
    conv = masked_conv2d(conv, 64, (1, 1), type='b', nonlinearity=rectify, pad='same')
    conv = masked_conv2d(conv, n_outputs, (1, 1), type='b', W=init.GlorotUniform(), nonlinearity=linear, pad='same')
    out = conv
    return inp, out

if __name__ == '__main__':
    import theano
    import theano.tensor as T
    inp = layers.InputLayer((None, 1, 32, 32))
    conv = inp
    conv = masked_conv2d(conv, 1, (1, 1), type='a', W=init.Constant(1.), nonlinearity=linear, pad='same')
    out = conv
    X = T.tensor4()
    y = layers.get_output(out, X)
    fn = theano.function([X], y)
    x, _ = np.indices((32, 32))
    x = x[np.newaxis, np.newaxis, :, :]
    print(x[0, 0, 0:10, 0:10])
    x = floatX(x)
    y = fn(x)
    print(x.shape, y.shape)
    print(y[0, 0, 0:10, 0:10])
