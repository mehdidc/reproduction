import numpy as np
from skimage.io import imsave

import theano
import theano.tensor as T

from lasagne import objectives
from lasagne import layers
from lasagne.updates import adam
from lasagne import nonlinearities

from model import build_model_pixelcnn, floatX
from helpers import disp
from helpers import sample_multinomial
from helpers import generate
from helpers import categ
from helpers import softmax
from helpers import floatX
from helpers import ColorDiscretizer, color_discretization

def run():
    batch_size = 128
    train_X = load_data()
    nb_channels = 4 # level of discretization
    print('Discretizing...')
    centers = color_discretization(train_X, nb_channels)
    print('Centers : {}'.format(centers))

    color_discretizater = ColorDiscretizer(centers=centers)
    color_discretizater.fit(train_X)
    train_X = color_discretizater.transform(train_X)

    train_X = categ(train_X, D=nb_channels)
    train_X = train_X.transpose((0, 3, 1, 2))
    
    # train_X has shape (nb_examples, nb_channels, h, w)
    print('data has shape : {}'.format(train_X.shape))
    height, width = train_X.shape[2:]
    input_shape = (None, nb_channels, height, width)
    inp, out = build_model_pixelcnn(
            input_shape=input_shape, 
            n_outputs=nb_channels)
    out = layers.NonlinearityLayer(out, softmax)
    
    print('Nb of params : {}'.format(layers.count_params(out)))
    X = T.tensor4() # (nb_examples, nb_channels, h, w)
    y = layers.get_output(out, X) # (nb_examples, nb_channels, h, w)
    
    X_ = X.argmax(axis=1).flatten() #(nb_examples * h * w,)
    y_ = y.transpose((1, 0, 2, 3)).flatten(2).T # (nb_examples*h*w, nb_channels)
    loss = objectives.categorical_crossentropy(y_, X_).mean()

    params = layers.get_all_params(out)
    updates = adam(loss, learning_rate=1e-3, params=params)

    print('compiling functions...')
    
    train = theano.function([X], loss, updates=updates)
    predict = theano.function([X], y)
    print(predict(train_X).shape)
    for epoch in range(10000):
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]

        avg_loss = 0.
        nb = 0
        print('training...')
        for i in range(0, len(train_X), batch_size):
            x = train_X[i:i+batch_size]
            loss_x = train(x)
            avg_loss += loss_x * len(x)
            nb += len(x)
        avg_loss /= nb
        print('train loss: {}'.format(avg_loss))
        if epoch % 100 == 0:
            #x = predict(train_X[0:9])
            #x = x.argmax(axis=1)
            #x = color_discretizater.inverse_transform(x)
            #x = disp(x, border=1, bordercolor=(1,0,0))
            #imsave('rec.png', x)
            print('generate...')
            x = np.zeros((100, nb_channels, height, width))
            x = generate(x, predict_fn=predict, sample_fn=sample_multinomial)
            x = color_discretizater.inverse_transform(x)
            x = disp(x, border=1, bordercolor=(1,0,0))
            imsave('out.png', x)

def load_data():
    from lasagnekit.datasets.mnist import MNIST
    data = MNIST()
    data.load()
    X = data.X
    X = X.reshape((X.shape[0], 1, 28, 28))
    return X[0:100]

if __name__ == '__main__':
    np.random.seed(42)
    run()
