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
    from caffezoo.googlenet import GoogleNet
    from caffezoo.vgg import VGG
    from lasagne import layers, updates
    import theano.tensor as T
    from helpers import load_word_embedding
    import numpy as np
    from lasagnekit.easy import (BatchOptimizer, build_batch_iterator,
                                 InputOutputMapping)
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.datasets.imagenet import ImageNet
    import pickle
    from collections import OrderedDict

    # imagenet data
    batch_size = 10
    size = (224, 224)
    imagenet = ImageNet(size=size, nb=batch_size)

    # word model
    logger.info("Loading word2vec...")
    word2vec = load_word_embedding(word2vec_filename)
    word2int = {word: i for i, word in enumerate(word2vec.keys())}  # NOQA
    int2word = {i: word for i, word in enumerate(word2vec.keys())}  # NOQA
    int2vec = np.array([word2vec[int2word[i]]
                        for i in range(len(word2vec))])
    int2vec /= np.sqrt((int2vec**2).sum(axis=1))[:, None]  # unit norm

    size_embedding = len(word2vec.values()[0])
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
    # loss function

    margin = 0.1

    def loss_function(model, tensors):
        img = tensors["img"]
        true_word = tensors["trueword"]
        false_word = tensors["falseword"]

        predicted_word = layers.get_output(output_layer, img)
        norm = (predicted_word**2).sum(axis=1).dimshuffle(0, 'x')
        predicted_word = predicted_word / norm

        true_term = (predicted_word * true_word).sum(axis=1)
        false_term = (predicted_word * false_word).sum(axis=1)

        return T.maximum(0, margin - true_term + false_term).sum()

    def transform(batch_index, batch_slice, tensors):
        imagenet.load()
        words = [pick_word(syn) for syn in imagenet.y]
        valid_samples = [i for i, w in enumerate(words)
                         if w is not None]
        words = [words[i] for i in valid_samples]
        words = [word2int[word] for word in words]

        tensors["img"] = visual_model_base.preprocess(imagenet.X[valid_samples])
        tensors["img"] = tensors["img"].astype(np.float32)
        tensors["trueword"] = embed(words).astype(np.float32)
        tensors["falseword"] = embed(pick_false_words(words)).astype(np.float32)
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
        return np.random.randint(0, len(int2vec), size=(len(words),))

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
                          params=["img"])
    )

    batch_optimizer = BatchOptimizer(
            verbose=2,
            max_nb_epochs=30000,
            optimization_procedure=(updates.adam, {"learning_rate": 0.001})
    )
    batch_iterator = build_batch_iterator(transform)

    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_iterator=batch_iterator,
                      batch_optimizer=batch_optimizer)

    logging.info("Training...")
    dummy = np.zeros((1, 1))
    capsule.fit(img=dummy, trueword=dummy, falseword=dummy)
    logging.info("Saving...")
    layers = layers.get_all_layers(output_layer)
    net = {layer.name: layer for layer in layers}
    with open("out/model.pkl", "w") as fd:
        pickle.dump(net, fd)
    logger.info("Ok")
