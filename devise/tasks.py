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
    from lasagne import layers
    import theano.tensor as T
    from helpers import load_word_embedding, size_embedding
    import numpy as np
    from lasagnekit.easy import BatchOptimizer, build_batch_iterator, InputOutputMapping
    from lasagnekit.nnet.capsule import Capsule

    from collections import OrderedDict
    # word model
    logger.info("Loading word2vec...")
    word2vec = load_word_embedding(word2vec_filename)
    word2int = {word: i for i, word in enumerate(word2vec.keys())} #NOQA
    int2word = {i: word for i, word in enumerate(word2vec.keys())} #NOQA
    int2vec = np.array([word2vec[int2word[i]]
                        for i in range(len(word2vec))])
    # visual model
    logger.info("Loading visual model...")
    googlenet = GoogleNet(layer_names=["input"], input_size=(224, 224))
    googlenet._load()
    base = googlenet._net
    model = build_visual_model(base,
                               input_layer_name="input",
                               repr_layer_name="pool1/norm1",
                               size_embedding=size_embedding(word2vec))
    input_layer, output_layer = model
    model = InputOutputMapping([input_layer], [output_layer])
    # loss function

    margin = 0.1

    def loss_function(model, tensors):
        img = tensors["img"]
        true_word = tensors["trueword"]
        false_word = tensors["falseword"]

        predicted_word = layers.get_output(output_layer, img.input_var)
        true_term = (predicted_word * true_word).sum(axis=1)
        false_term = (predicted_word * false_word).sum(axis=1)

        return T.maximum(0, margin - true_term + false_term).sum()

    def transform(batch_index, batch_slice, tensors):
        tensors["img"] = tensors["img"][batch_slice]
        tensors["trueword"] = embed(tensors["trueword"][batch_slice])
        tensors["falseword"] = embed(pick_false_words(tensors["trueword"]))
        return tensors

    def embed(wordint):
        return int2vec[wordint]

    def pick_false_words(words):
        return np.random.randint(0, len(int2vec), size=(len(words),))

    print(embed(pick_false_words([0])).shape)

    logging.info("Constructing the model...")
    input_variables = OrderedDict(
        img=dict(tensor_type=T.tensor4),
        trueword=dict(tensor_type=T.ivector),
        falseword=dict(tensor_type=T.ivector)
    )

    functions = dict(
             predict=dict(get_output=lambda model, img: layers.get_output(output_layer, img),
                          params=["img"])
    )

    batch_optimizer = BatchOptimizer(verbose=2)
    batch_iterator = build_batch_iterator(transform)

    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer)

    logging.info("Training")
