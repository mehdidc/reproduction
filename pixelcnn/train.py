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
from helpers import categ
from helpers import softmax
from helpers import floatX
from helpers import ColorDiscretizerPerChannel, ColorDiscretizerRound, color_discretization, ColorDiscretizerJoint
from helpers import random_crop

def run(data, folder='.'):
    train_X = data
    batch_size = 64
    nb_layers = 2
    print('Discretizing...')
    use_centers = True # False if  you want it like the original paper
    joint_centers = True # False if you want it like the original paper
    if use_centers:
        nb_centers = 256
        centers = color_discretization(train_X[0:100], nb_centers)
        print('Centers : {}'.format(centers))
        if joint_centers:
            color_discretizer = ColorDiscretizerJoint(centers=centers, batch_size=128)
            nb_outputs = nb_centers
            generate_fn = generate_with_joint_colors
            mask_channels = False
        else:
            color_discretizer = ColorDiscretizerPerChannel(centers=centers, batch_size=128)
            nb_outputs = nb_centers
            generate_fn = generate
            mask_channels = True
    else:
        color_discretizer = ColorDiscretizerRound()
        nb_outputs = 256
        generate_fn = generate
        mask_channels = True

    color_discretizer.fit(train_X)
    train_X = color_discretizer.transform(train_X)
    train_X = floatX(train_X)

    # train_X has shape (nb_examples, h, w)
    print('data has shape : {}'.format(train_X.shape))

    img = disp(color_discretizer.inverse_transform(train_X[0:100]), border=1, bordercolor=(1,0,0))
    imsave(folder+'/real.png', img)
 
     
    if joint_centers:
        height, width = train_X.shape[1:]
        nb_channels = nb_outputs
    else:
        nb_channels, height, width = train_X.shape[1:]

    input_shape = (None, nb_channels, height, width)
    inp, out = build_model_pixelcnn(
            input_shape=input_shape, 
            nb_outputs=nb_outputs,
            nb_layers=nb_layers,
            dim=21,
            mask_channels=mask_channels)
    out = layers.NonlinearityLayer(out, softmax)
    
    print('Nb of params : {}'.format(layers.count_params(out)))

    if joint_centers:
        print(out.output_shape)
        #nb_channels=1 and nb_outputs = nb_centers here
        X = T.tensor4() # (nb_examples, nb_outputs, h, w)
        y = layers.get_output(out, X) # (nb_examples, nb_outputs, nb_channels=1, h, w)
        X_ = X.argmax(axis=1).flatten() #(nb_examples * h * w,)
        y_ = y[:, :, 0, :, :] #(nb_examples, nb_outputs, h, w)
        y_ = y_.transpose((1, 0, 2, 3)).flatten(2).T # (nb_examples*h*w, nb_outputs)
        loss = objectives.categorical_crossentropy(y_, X_).mean()
    else:
        X = T.tensor4() # (nb_examples, nb_channels, h, w)
        y = layers.get_output(out, X/255.) # (nb_examples, nb_outputs, nb_channels, h, w)
        X_ = X.transpose((0, 2, 3, 1)).flatten() #(nb_examples * h * w * nb_channels,)
        X_ = T.cast(X_, 'int32')
        y_ = y.transpose((0, 3, 4, 2, 1)).reshape((-1, nb_outputs)) # (nb_examples * h * w * nb_channels, nb_outputs)
        loss = objectives.categorical_crossentropy(y_, X_).mean()

    params = layers.get_all_params(out)
    updates = adam(loss, learning_rate=1e-4, params=params)

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
            if joint_centers:
                x = categ(x, D=nb_outputs)
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
        if epoch % 100 == 0:
            print('generate...')
            t = time.time()
            x = np.zeros((9, nb_channels, height, width))
            x = generate_fn(x, predict_fn=predict, sample_fn=sample_multinomial)
            x = color_discretizer.inverse_transform(x)
            x = disp(x, border=1, bordercolor=(1,0,0))
            imsave(folder+'/out{:05d}.png'.format(epoch), x)
            dt = time.time() - t
            print('generation duration : {:.5f}s'.format(dt))

def generate(X, predict_fn, sample_fn=sample_multinomial):
    X = floatX(X)
    out = np.empty_like(X)
    nb_examples, nb_channels, h, w = X.shape
    for y in range(h):
        for x in range(w):
            p = predict_fn(X)
            for channel in range(nb_channels):
                sample = sample_fn(p[:, :, channel, y, x])
                print(sample)
                X[:, channel, y, x] = sample
                out[:, channel, y, x] = sample
    return out

def generate_with_joint_colors(X, predict_fn, sample_fn=sample_multinomial):
    X = floatX(X)
    out = np.empty((X.shape[0],) + (X.shape[2:]), dtype='int32')
    nb, nb_channels, h, w = X.shape
    for y in range(h):
        for x in range(w):
            p = predict_fn(X) # (nb_examples, nb_outputs, nb_channels=1, h, w)
            sample = sample_fn(p[:, :, 0, y, x]) #nb_channels=1 for joint colors
            X[:, :, y, x] = categ(sample, D=nb_channels)
            out[:, y, x] = sample
    return out

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
        data = imap(lambda x:x if len(x.shape) == 3 else x[:, :, np.newaxis], data) # add a channel axis for grayscale images
        data = imap(partial(resize, output_shape=(16, 16), preserve_range=True), data) # resize the images
        #data = imap(partial(random_crop, shape=(8, 8)), data) # crop the images 
        data = imap(lambda X:X.transpose((2, 0, 1)), data) # make the channel at the beginning (channels, h, w)
        data = list(data)
        data = np.array(data)
        data = floatX(data)
    run(data=data, folder=args['FOLDER'])
