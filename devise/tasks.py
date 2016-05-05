# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
from invoke import task

import sys
import logging

logger = logging.getLogger("tasks.py")
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@task
def train(word2vec_filename="data/glove.6B.50d.pkl"):
    from model import build_visual_model
    from caffezoo.googlenet import GoogleNet  # NOQA
    from caffezoo.vgg import VGG
    from lasagne import layers, updates
    import theano.tensor as T
    from helpers import load_word_embedding
    import numpy as np
    from lasagnekit.easy import build_batch_iterator
    from lasagnekit.easy import (
            make_batch_optimizer, InputOutputMapping, exp_moving_avg,
            get_stat)
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.datasets.imagenet import ImageNet
    import pickle
    from collections import OrderedDict
    import theano

    # imagenet data
    batch_size = 10
    size = (224, 224)
    imagenet = ImageNet(size=size, nb=batch_size, crop=True)
    imagenet.rng = np.random.RandomState(2)

    batch_size_valid = 1000
    imagenet_valid = ImageNet(size=size, nb=batch_size_valid, crop=True)
    imagenet_valid.rng = np.random.RandomState(8)
    imagenet_valid.load()

    # word model
    logger.info("Loading word2vec...")
    word2vec = load_word_embedding(word2vec_filename)

    word2int = {word: i for i, word in enumerate(word2vec.keys())}  # NOQA
    int2word = {i: word for i, word in enumerate(word2vec.keys())}  # NOQA
    int2vec = np.array([word2vec[int2word[i]]
                        for i in range(len(word2vec))])

    def norm(X):
        return np.sqrt((X**2).sum(axis=1))[:, None]
    int2vec /= norm(int2vec)

    size_embedding = len(word2vec.values()[0])
    imagenet.false_words = set(np.random.randint(0, len(int2vec), size=(100,)))
    # visual model
    logger.info("Loading visual model...")
    cls = VGG
    visual_model_base = cls(layer_names=["input"], input_size=size,
                            resize=False)
    visual_model_base._load()
    model = build_visual_model(visual_model_base._net,
                               input_layer_name="input",
                               repr_layer_name="fc6",
                               size_embedding=size_embedding)
    input_layer, output_layer = model
    model = InputOutputMapping([input_layer], [output_layer])

    def get_all_params(**kwargs):
        return output_layer.get_params(**kwargs)

    model.get_all_params = get_all_params
    # loss function

    margin = 0.1

    def loss_function(model, tensors):
        img = tensors["img"]
        true_word = tensors["trueword"]
        false_word = tensors["falseword"]

        predicted_word = layers.get_output(output_layer, img)
        norm = T.sqrt((predicted_word**2).sum(axis=1).dimshuffle(0, 'x'))
        predicted_word = predicted_word / norm
        
        #return ((predicted_word - true_word) ** 2).sum()

        true_term = (predicted_word * true_word).sum(axis=1)
        false_term = (predicted_word * false_word).sum(axis=1)
        return T.maximum(0, margin - true_term + false_term).sum()
    imagenet.counter = 0
    imagenet.load()

    def closest_words(sample_vecs, nb_words=1):
        s = np.argsort(np.dot(sample_vecs, int2vec.T), axis=1)
        s = s[:, int2vec.shape[0] - nb_words:int2vec.shape[0]]
        W = []
        for a in s:
            W.append(map(lambda i: int2word[i], a))
        return W

    def transform(batch_index, batch_slice, tensors):
        imagenet.load()
        words = [pick_word(syn) for _, syn in imagenet.y]
        imagenet.false_words |= set([word2int[w]
                                     for w in words if w is not None])
        valid_samples = [i for i, w in enumerate(words)
                         if w is not None]
        words = [words[i] for i in valid_samples]
        words = [word2int[word] for word in words]

        tensors["img"] = visual_model_base.preprocess(
            imagenet.X[valid_samples])
        tensors["img"] = tensors["img"].astype(np.float32)
        tensors["trueword"] = embed(words).astype(np.float32)
        tensors["falseword"] = embed(
            pick_false_words(words)).astype(np.float32)
        return tensors

    def pick_word(syn):
        W = [w for s in syn for w in s.split()]
        W = filter(lambda w: w in word2vec, W)
        if len(W) > 0:
            return W[0]
        else:
            return None

    def embed(wordint):
        return int2vec[wordint]

    def pick_false_words(words):
        return np.random.choice(list(imagenet.false_words), size=(len(words),))

    logging.info("Constructing the model...")
    input_variables = OrderedDict(
        img=dict(tensor_type=T.tensor4),
        trueword=dict(tensor_type=T.matrix),
        falseword=dict(tensor_type=T.matrix)
    )

    def pred(model, img):
        return layers.get_output(output_layer, img)

    functions = dict(
        predict=dict(get_output=pred,
                     params=["img"]),
    )

    def update_status(batch_optimizer, status):

        t = status["epoch"]

        cur_lr = lr.get_value()

        if lr_decay_method == "exp":
            new_lr = cur_lr * (1 - lr_decay)
        elif lr_decay_method == "lin":
            new_lr = initial_lr / (1 + t)
        elif lr_decay_method == "sqrt":
            new_lr = initial_lr / np.sqrt(1 + t)
        else:
            new_lr = cur_lr

        new_lr = np.array(new_lr, dtype="float32")
        lr.set_value(new_lr)

        status = exp_moving_avg(batch_optimizer.stats,
                                "loss_train", "avg_loss_train",
                                status)

        if t % 10 == 0:
            report_(status)

        return status

    def report_(status):
        fig = plt.figure()
        plt.plot(get_stat("avg_loss_train", capsule.batch_optimizer.stats),
                 label="train")
        plt.savefig("out/loss.png")
        plt.close(fig)

        n = 10
        s = np.random.randint(0, batch_size_valid, size=n)
        # s = np.arange(0, n)
        images = imagenet_valid.X[s]
        true_labels = [imagenet_valid.y[i] for i in s]
        E = capsule.predict(visual_model_base.preprocess(images))
        E /= norm(E)
        closest_to_E = closest_words(E, nb_words=5)
        fig = plt.figure(figsize=(10, 10))
        for i in range(n):
            plt.subplot(n, 1, i + 1)
            plt.axis('off')
            plt.imshow(images[i])
            true = true_labels[i][1][0]
            pred = closest_to_E[i]
            pred = [p.decode("utf8") for p in pred]
            true = true.decode("utf8")
            title = u"Closest : {}, True : {}".format(",".join(pred), true)
            plt.title(title)
        plt.savefig("out/pred{}.png".format(status["epoch"]))
        plt.close(fig)

    lr_decay_method = "sqrt"
    initial_lr = 0.001
    lr_decay = 0.0000001
    lr = theano.shared(np.array(initial_lr, dtype=np.float32))
    momentum = 0.9
    algo = updates.rmsprop
    params = {"learning_rate": lr}

    if algo in (updates.momentum, updates.nesterov_momentum):
        params["momentum"] = momentum
    optim = (algo, params)
    batch_optimizer = make_batch_optimizer(
        update_status,
        max_nb_epochs=100000,
        optimization_procedure=optim,
        patience_stat='avg_loss_train_fix',
        patience_nb_epochs=800,
        min_nb_epochs=1000,
        batch_size=batch_size,
        verbose=1)

    batch_iterator = build_batch_iterator(transform)

    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_iterator=batch_iterator,
                      batch_optimizer=batch_optimizer)
    logging.info("Training...")
    dummy = np.zeros((1, 1))
    try:
        capsule.fit(img=dummy, trueword=dummy, falseword=dummy)
    except KeyboardInterrupt:
        pass
    logging.info("Saving...")
    layers = layers.get_all_layers(output_layer)
    net = {layer.name: layer for layer in layers}
    with open("out/model.pkl", "w") as fd:
        pickle.dump(net, fd)
    logger.info("Ok")
