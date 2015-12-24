from invoke import task
from model import build_model_lasagne, build_model_keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
import numpy as np
import sys


@task
def train(filename):

    with open(filename, "r") as fd:
        lines = list(fd.readlines())
    #lines = [
    #    "1i was there",
    #    "2she was here",
    #    "3i want there",
    #    "4she was like that",
    #    "5she likes him",
    #    "6he likes her",
    #]
    text = [c for l in lines for c in l]
    chars = list(set(text))
    char_to_int = {c: i + 1 for i, c in enumerate(chars)}
    int_to_char = {i + 1: c for i, c in enumerate(chars)}
    char_to_int[0] = 0
    int_to_char[0] = 0

    def word2int(w):
        return char_to_int[w]

    def int2word(i):
        return int_to_char[i]

    maxlen = max(map(len, lines)) + 2
    corpus = lines

    words = set(word for sentence in corpus for word in sentence)
    nb_words = len(words) + 1

    def preprocess(corpus, pad=False):
        corpus = [map(word2int, sentence) for sentence in corpus]
        if pad:
            corpus = pad_sequences_(corpus, maxlen=maxlen - 2)
        corpus = map(lambda s: ([0] + s + [0]), corpus)
        corpus = [label_binarize(sentence, np.arange(nb_words))
                  for sentence in corpus]
        return corpus

    def pad_sequences_(t, maxlen=None):
        if maxlen is None:
            maxlen = max(map(len, t))
        t = [el + [0] * (maxlen - len(el)) if len(el) < maxlen
             else el[0:maxlen]
             for el in t]
        return t

    def deprocess(corpus):
        corpus = [np.array(sentence).argmax(axis=1).tolist()
                  for sentence in corpus]
        print(np.array(corpus).shape)
        corpus = [map(int2word, sentence) for sentence in corpus]
        return corpus

    def show(corpus):
        for sentence in corpus:
            sentence = map(lambda c: "0" if c == 0 else c, sentence)
            sys.stdout.write("".join(sentence))

    corpus = preprocess(corpus, pad=True)
    prev, cur, next_ = corpus[0:-2], corpus[1:-1], corpus[2:]
    backend = "lasagne"
    if backend == "keras":
        train_keras_(maxlen, nb_words, cur, prev, next_)
    elif backend == "lasagne":
        train_lasagne_(
                maxlen, nb_words, cur, prev, next_,
                preprocess, deprocess, show)


def train_keras_(maxlen, nb_words, cur, prev, next_):
    model = build_model_keras(
            input_length=maxlen,
            nb_words=nb_words, hidden=512)
    loss = {'prev_output': 'categorical_crossentropy',
            'next_output': 'categorical_crossentropy'}
    model.compile(
            optimizer='rmsprop',
            loss=loss
    )
    params = {'input': cur, 'prev_output': prev, 'next_output': next_}

    json_string = model.to_json()
    open('out/arch.json', 'w').write(json_string)
    for i in range(1000):
        model.fit(params, nb_epoch=1)
        model.save_weights('out/weights.h5', overwrite=True)


def train_lasagne_(maxlen, nb_words, cur, prev, next_, preprocess, deprocess, show):
    from lasagne import updates, layers
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.easy import make_batch_optimizer, InputOutputMapping
    import theano.tensor as T
    from collections import OrderedDict
    import theano
    np.random.seed(2)
    model_layers = build_model_lasagne(input_length=maxlen,
                                       prev_length=maxlen - 1,
                                       next_length=maxlen - 1,
                                       nb_words=nb_words, hidden=512)
    print(model_layers.keys())

    def categorical_crossentropy_(p, y):
        p = p.reshape((p.shape[0] * p.shape[1], p.shape[2]))
        y = y.reshape((y.shape[0] * y.shape[1],))
        return -T.log(p[T.arange(p.shape[0]), y])

    def loss_function(model, tensors):
        input_ = tensors["input"]
        prev = tensors["prev"]
        next_ = tensors["next"]

        prev_output = tensors["prev_output"]
        next_output = tensors["next_output"]
        pred_prev, pred_next = model.get_output(input_, prev, next_)
        loss_prev = categorical_crossentropy_(pred_prev, prev_output.argmax(axis=2))
        loss_next = categorical_crossentropy_(pred_next, next_output.argmax(axis=2))

        #loss_prev = ((pred_prev - prev_output) ** 2).sum(axis=2).sum(axis=1).mean()
        #loss_next = ((pred_next - next_output) ** 2).sum(axis=2).sum(axis=1).mean()
        return (loss_prev + loss_next).mean()

    variables = OrderedDict({
        "input": dict(tensor_type=T.tensor3),
        "prev": dict(tensor_type=T.tensor3),
        "next": dict(tensor_type=T.tensor3),
        "prev_output": dict(tensor_type=T.tensor3),
        "next_output": dict(tensor_type=T.tensor3),
    })

    def get_code(model, input_):
        return layers.get_output(model_layers["code"], input_)
    functions = dict(
            get_code=dict(get_output=get_code, params=["input"]),
    )

    learning_rate = theano.shared(np.array(0.001, dtype="float32"))

    def update_status(batch_optimizer, status):
        #cur = ["import sys\n"]
        # cur = ["1i was there"]
        gen = ["import sys\n"]
        for i in range(10):
            gen = generate_lasagne_(capsule.get_code, capsule.get_next,
                                    preprocess, deprocess,
                                    gen,
                                    max_length=maxlen,
                                    rng=np.random)
            show(gen)
        #v = learning_rate.get_value() * 0.98
        #v = np.array(v, dtype="float32")
        #learning_rate.set_value(v)
        print("")
        return status
    batch_optimizer = make_batch_optimizer(
            update_status,
            max_nb_epochs=10000,
            optimization_procedure=(updates.adam, {'learning_rate': learning_rate}),
            verbose=1)

    inputs = [model_layers["input"], model_layers["prev"], model_layers["next"]]
    outputs = [model_layers["prev_output"], model_layers["next_output"]]
    model = InputOutputMapping(inputs, outputs)
    capsule = Capsule(variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer)

    t_code = T.matrix()
    t_next = capsule.v_tensors["next"]
    v = {
        model_layers["code"]: t_code,
        model_layers["next"]: t_next
    }
    capsule.get_next = theano.function(
        [t_code, t_next],
        layers.get_output(model_layers["next_output"], v))
    # creating dataset
    prev_in = [p[0:-1] for p in prev]
    prev_out = [p[1:] for p in prev]
    next_in = [n[0:-1] for n in next_]
    next_out = [n[1:] for n in next_]
    #all_variables = [
    #    cur, prev, next_, prev_out, next_out
    #]
    #cur, prev, next_, prev_out, next_out = shuffle(*all_variables)

    show(deprocess(cur))
    print("")
    show(deprocess(prev_in))
    print("")
    show(deprocess(next_in))
    print("")
    show(deprocess(prev_out))
    print("")
    show(deprocess(next_out))

    t = OrderedDict({
        "input": cur,
        "prev": prev_in,
        "next": next_in,
        "prev_output": prev_out,
        "next_output": next_out,
    })
    t = OrderedDict(t)
    capsule.fit(**t)


def generate_lasagne_(get_code, get_next,
                      preprocess, deprocess,
                      cur,
                      max_length=10,
                      rng=np.random):
    cur_ = preprocess(cur, pad=True)
    cur_ = np.array(cur_)
    nb_words = cur_.shape[2]
    code = get_code(cur_)
    gen = np.zeros((len(cur_), 1, nb_words)).astype(np.float32)
    gen[:, :, 0] = 1.

    for i in range(max_length):
        probas = get_next(code, gen)
        probas = probas[:, -1, :]
        next_gen = []
        for proba in probas:
            #word_idx = rng.multinomial(1, proba).argmax()
            word_idx = proba.argmax()
            one_hot = [0] * nb_words
            one_hot[word_idx] = 1.
            next_gen.append(one_hot)
        next_gen = np.array(next_gen, dtype="float32")[:, None, :]
        gen = np.concatenate((gen, next_gen), axis=1)
    print(gen.shape)
    return deprocess(gen)
