import time
import pandas as pd
import numpy as np
from skimage.io import imsave

import theano
import theano.tensor as T

from lasagne import objectives
from lasagne import layers
from lasagne.layers import batch_norm
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

def run(data, folder='.'):
    train_X = data
    batch_size = 64
    nb_channels = 3 # level of discretization
    nb_layers = 8

    print('Discretizing...')
    centers = color_discretization(train_X[0:100], nb_channels)
    print('Centers : {}'.format(centers))

    color_discretizater = ColorDiscretizer(centers=centers, batch_size=128)
    color_discretizater.fit(train_X)
    train_X = color_discretizater.transform(train_X)
    
    # train_X has shape (nb_examples, h, w)
    print('data has shape : {}'.format(train_X.shape))
    height, width = train_X.shape[1:]
    input_shape = (None, nb_channels, height, width)
    inp, out = build_model_pixelcnn(
            input_shape=input_shape, 
            n_outputs=nb_channels,
            nb_layers=nb_layers)
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
    losses = []
    for epoch in range(10000):
        t = time.time()
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        avg_loss = 0.
        nb = 0
        print('training...')
        for i in range(0, len(train_X), batch_size):
            x = train_X[i:i+batch_size]
            x = categ(x, D=nb_channels)
            x = x.transpose((0, 3, 1, 2))
            loss_x = train(x)
            avg_loss += loss_x * len(x)
            nb += len(x)
        avg_loss /= nb
        dt = time.time() - t
        print('train loss: {}'.format(avg_loss))
        print('train duration : {:.5f}s'.format(dt))
        losses.append(avg_loss)
        pd.Series(losses).to_csv(folder+'/loss.csv')
        if epoch % 10 == 0:
            print('generate...')
            t = time.time()
            x = np.zeros((100, nb_channels, height, width))
            x = generate(x, predict_fn=predict, sample_fn=sample_multinomial)
            x = color_discretizater.inverse_transform(x)
            x = disp(x, border=1, bordercolor=(1,0,0))
            imsave(folder+'/out{:05d}.png'.format(epoch), x)
            dt = time.time() - t
            print('generation duration : {:.5f}s'.format(dt))
 

if __name__ == '__main__':
    from docopt import docopt
    from datakit.imagecollection import load
    import datakit
    import glob
    from itertools import imap
    from itertools import cycle
    from functools import partial
    from skimage.transform import resize
    import random
    doc = """
    Usage: train.py PATTERN FOLDER
    """
    args = docopt(doc)
    np.random.seed(42)
    datasets = ['mnist', 'cifar']
    if args['PATTERN'] in datasets:
        dataset = getattr(datakit, args['PATTERN'])
        data = dataset.load()
        data = data['train']['X']
    else: 
        filelist = glob.glob(args['PATTERN'])
        random.shuffle(filelist)
        filelist = filelist[0:100]
        data = load(filelist, buffer_size=len(filelist))
        data = imap(partial(resize, output_shape=(64, 64), preserve_range=True), data)
        data = imap(lambda X:X.transpose((2, 0, 1)), data)
        data = list(data)
        data = np.array(data)
        data = floatX(data)
    run(data=data, folder=args['FOLDER'])
