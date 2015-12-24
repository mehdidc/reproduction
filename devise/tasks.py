from invoke import task

import sys
import logging

from controller import launch, Controller

from lasagnekit.easy import BatchOptimizer

logger = logging.getLogger("tasks.py")
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class MyBatchOptimizer(BatchOptimizer):
    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(MyBatchOptimizer, self).iter_update(
                epoch, nb_batches, iter_update_batch)
        self.controller_values["controller"].handle()
        return status


@task
def train(word2vec_filename="data/glove.6B.50d.pkl"):
    from model import build_visual_model
    from caffezoo.googlenet import GoogleNet  # NOQA
    from caffezoo.vgg import VGG
    from lasagne import layers, updates
    import theano.tensor as T
    from helpers import load_word_embedding
    import numpy as np
    from lasagnekit.easy import (build_batch_iterator,
                                 InputOutputMapping)
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.datasets.imagenet import ImageNet
    import pickle
    from collections import OrderedDict

    # imagenet data
    batch_size = 10
    size = (224, 224)
    imagenet = ImageNet(size=size, nb=batch_size, crop=True)
    imagenet.rng = np.random.RandomState(2)

    # word model
    logger.info("Loading word2vec...")
    word2vec = load_word_embedding(word2vec_filename)
    word2int = {word: i for i, word in enumerate(word2vec.keys())}  # NOQA
    int2word = {i: word for i, word in enumerate(word2vec.keys())}  # NOQA
    int2vec = np.array([word2vec[int2word[i]]
                        for i in range(len(word2vec))])
    int2vec /= np.sqrt((int2vec**2).sum(axis=1))[:, None]  # unit norm

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
        #norm = (predicted_word**2).sum(axis=1).dimshuffle(0, 'x')
        #predicted_word = predicted_word / norm

        true_term = (predicted_word * true_word).sum(axis=1)
        false_term = (predicted_word * false_word).sum(axis=1)
        
        return ((predicted_word - true_word ) ** 2).sum()
        return T.maximum(0, margin - true_term + false_term).sum()
    imagenet.counter = 0
    imagenet.load()
    def transform(batch_index, batch_slice, tensors):
        #imagenet.load()
        imagenet.counter += 1

        if imagenet.counter % 1000 == 0:
            imagenet.counter = 0
            #imagenet.rng = np.random.RandomState(2)
            imagenet.load()
            print("Switch back to 0")

        words = [pick_word(syn) for syn in imagenet.y]
        imagenet.false_words |= set([word2int[w] for w in words if w is not None])
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
        return np.random.choice(list(imagenet.false_words), size=(len(words),))
        #return np.random.randint(0, len(int2vec), size=(len(words),))

    logging.info("Constructing the model...")
    input_variables = OrderedDict(
        img=dict(tensor_type=T.tensor4),
        trueword=dict(tensor_type=T.matrix),
        falseword=dict(tensor_type=T.matrix)
    )

    def pred(model, img):
        return layers.get_output(output_layer, img)

    def grad(model, img, trueword, falseword):
        tensors = dict(
            img=img,
            trueword=trueword,
            falseword=falseword
        )
        L = loss_function(model, tensors)


    functions = dict(
             predict=dict(get_output=pred,
                          params=["img"]),
    )
    controller = Controller()
    controller_values = dict(
            last_grad_layers=None,
            learning_rates=None,
            momentums=None,
            controller=controller
    )
    learning_rate = 0.0001
    batch_optimizer = MyBatchOptimizer(
            verbose=2,
            max_nb_epochs=300000,
            batch_size=batch_size,
            optimization_procedure=(updates.rmsprop,
                                    {"learning_rate": learning_rate})
    )

    batch_optimizer.controller_values = controller_values
    batch_iterator = build_batch_iterator(transform)

    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_iterator=batch_iterator,
                      batch_optimizer=batch_optimizer,
                      store_grads_params=True)
    controller_values["last_grad_layers"] = capsule._grads
    launch(controller_values, name="DEVISE", register=False)
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
