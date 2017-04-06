import time
import pandas as pd
import numpy as np
from skimage.io import imsave

import theano
import theano.tensor as T

from lasagne import objectives
from lasagne import layers
from lasagne.layers import batch_norm
from lasagne.updates import adam, adadelta
from lasagne import nonlinearities

from model import build_model_pixelcnn, build_model_aa, build_model_aa_vertebrate, floatX
from helpers import disp
from helpers import sample_multinomial
from helpers import categ
from helpers import softmax
from helpers import floatX
from helpers import ColorDiscretizerPerChannel, ColorDiscretizerRound, color_discretization, ColorDiscretizerJoint
from helpers import random_crop
from helpers import mkdir_path

def run(data, folder='.'):
    mkdir_path(folder)
    mkdir_path(folder+'/rec')
    mkdir_path(folder+'/gen')
    train_X = data
    batch_size = 128
    nb_layers = 7
    print('Discretizing...')
    use_centers = True # False if  you want it like the original paper
    joint_centers = True # False if you want it like the original paper
    if use_centers:
        nb_centers = 100
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
    
    generate_fn = generate_refinement
    color_discretizer.fit(train_X)
    
    train_X = color_discretizer.transform(train_X)
    train_X = floatX(train_X)

    # train_X has shape (nb_examples, h, w)
    print('data has shape : {}'.format(train_X.shape))
    x = color_discretizer.inverse_transform(train_X[0:100])
    img = disp(x, border=1, bordercolor=(1,0,0))
    imsave(folder+'/real.png', img)
     
    if joint_centers:
        height, width = train_X.shape[1:]
        nb_channels = nb_outputs
    else:
        nb_channels, height, width = train_X.shape[1:]

    input_shape = (None, nb_channels, height, width)
    build_model = build_model_aa
    inp, out = build_model(
            input_shape=input_shape, 
            nb_outputs=nb_outputs,
            nb_layers=nb_layers,
            dim=128,
            mask_channels=mask_channels)
    out = layers.NonlinearityLayer(out, softmax)
    
    print('Nb of params : {}'.format(layers.count_params(out)))
    n_steps = 1
    if joint_centers:
        print(out.output_shape)
        #nb_channels=1 and nb_outputs = nb_centers here
        X = T.tensor4() # (nb_examples, nb_outputs, h, w)
        X_ = X.argmax(axis=1).flatten() #(nb_examples * h * w,)
        y = X
        for _ in range(n_steps):
            y = layers.get_output(out, y) # (nb_examples, nb_outputs, nb_channels=1, h, w)
            y_raw = y
            y = y[:, :, 0, :, :] #(nb_examples, nb_outputs, h, w)
        y = y.transpose((1, 0, 2, 3)).flatten(2).T # (nb_examples*h*w, nb_outputs)
        eps = 10e-8 
        y = T.clip(y, eps, 1 - eps)
        loss = objectives.categorical_crossentropy(y, X_).mean()
    else:
        X = T.tensor4() # (nb_examples, nb_channels, h, w)
        y = layers.get_output(out, X/train_X.max()) # (nb_examples, nb_outputs, nb_channels, h, w)
        y_raw = y
        X_ = X.transpose((0, 2, 3, 1)).flatten() #(nb_examples * h * w * nb_channels,)
        X_ = T.cast(X_, 'int32')
        y = y.transpose((0, 3, 4, 2, 1)).reshape((-1, nb_outputs)) # (nb_examples * h * w * nb_channels, nb_outputs)
        eps = 10e-8 
        y = T.clip(y, eps, 1 - eps)
        loss = objectives.categorical_crossentropy(y, X_).mean()

    params = layers.get_all_params(out)

    updates = adam(loss, learning_rate=1e-3, params=params)
    #updates = adadelta(loss, learning_rate=1, params=params)

    print('compiling functions...')
    
    train = theano.function([X], loss, updates=updates)
    predict = theano.function([X], y_raw)
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
            x_sample = x
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
            x = generate_fn(x, predict_fn=predict, sample_fn=sample_multinomial, transformer=color_discretizer)
            print(x.shape)
            x = color_discretizer.inverse_transform(x)
            x = disp(x, border=1, bordercolor=(1,0,0))
            imsave(folder+'/gen/out{:05d}.png'.format(epoch), x)

            xpred = predict(x_sample)
            xpred = xpred.argmax(axis=1)[:, 0]
            print(xpred.shape)
            xpred = color_discretizer.inverse_transform(xpred)
            xpred = disp(xpred, border=1, bordercolor=(1, 0, 0))
            imsave(folder+'/rec/rec{:05d}.png'.format(epoch), xpred)
            dt = time.time() - t

            print('generation duration : {:.5f}s'.format(dt))

            for lay in layers.get_all_layers(out):
                if not hasattr(lay, 'W'):
                    continue
                W = lay.W.get_value()
                if W.shape[1] == nb_outputs:
                    print(W.shape)

def generate(X, predict_fn, sample_fn=sample_multinomial):
    X = floatX(X)
    out = np.empty_like(X)
    nb_examples, nb_channels, h, w = X.shape
    for y in range(h):
        for x in range(w):
            for channel in range(nb_channels):
                p = predict_fn(X)
                sample = sample_fn(p[:, :, channel, y, x])
                X[:, channel, y, x] = sample
                out[:, channel, y, x] = sample
    return out

def quantize(X):
    mask = (X == X.max(axis=1, keepdims=True))
    X[mask] = 1
    X[~mask] = 0
    X = floatX(X)
    return X

def generate_refinement(X, predict_fn, sample_fn=sample_multinomial, transformer=None):
    X = np.random.uniform(size=X.shape)
    X = quantize(X)
    for _ in range(20):
        Xprev = X.copy()
        X = predict_fn(X)
        X = X[:, :, 0]
        # binarizing
        X = quantize(X)
        # sampling
        """
        X = X.transpose((0, 2, 3, 1))
        shape = X.shape
        X = X.reshape((X.shape[0]*X.shape[1]*X.shape[2], X.shape[3]))
        X = sample_fn(X)
        D = shape[3]
        X = categ(X, D=D)
        X = X.reshape(shape)
        X = X.transpose((0, 3, 1, 2))
        """
        score = np.abs(X-Xprev).mean()
        print(score)
        if score==0:
            break
    X = X.argmax(axis=1)
    return X

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
    from shutil import rmtree
    from docopt import docopt
    import glob
    from itertools import imap
    from itertools import cycle
    from functools import partial
    import random
    from skimage.transform import resize
    from datakit.image import pipeline_load
    from functools import partial
    from itertools import imap
    from datakit.helpers import expand_dict, dict_apply, minibatch
    doc = """
    Usage: train.py [--nb-examples=<int>] PATTERN FOLDER

    Options:
        -h help
    """
    args = docopt(doc)
    np.random.seed(42)
    datasets = ['mnist', 'cifar']
    if args['PATTERN'] in datasets:
        dataset = getattr(datakit, args['PATTERN'])
        data = dataset.load()
        data = data['train']['X']
        if args['--nb-examples']:
            nb_examples = int(args['--nb-examples'])
            data = data[0:nb_examples]
    else: 
        pipeline = [
            {"name": "imagefilelist", "params": {"pattern": args['PATTERN']}}
        ]
        if args['--nb-examples']:
            nb_examples = int(args['--nb-examples'])
            pipeline += [{"name": "limit", "params": {"nb": nb_examples}}]
        pipeline += [
            {"name": "shuffle", "params": {}},
            {"name": "imageread", "params": {}},
            {"name": "normalize_shape", "params": {}},
            {"name": "force_rgb", "params": {}},
            {"name": "resize", "params": {"shape": [32, 32]}},
            {"name": "order", "params": {"order": "th"}}
        ]
        nb_examples = len(list(pipeline_load(pipeline[0:1])))
        data = pipeline_load(pipeline)
        data = minibatch(data, batch_size=nb_examples)
        data = expand_dict(data)
        data = imap(partial(dict_apply, fn=floatX, cols=['X']), data)
        data = next(data)['X']
        data = np.array(data)
        data = floatX(data)
        print(data.shape)
    try:
        rmtree(args['FOLDER'])
    except Exception:
        pass
    run(data=data, folder=args['FOLDER'])
