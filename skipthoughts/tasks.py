from __future__ import print_function
import matplotlib as mpl
import os
mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
from invoke import task
import numpy as np
import sys
from sklearn.preprocessing import label_binarize
from model import build_model_skipthoughts, build_model_text_generation, skipthoughts_params
from helpers import softmax
from hp_toolkit.hp import Param, make_constant_param, instantiate_random, instantiate_default
from lightexperiments.light import Light


params = dict(
    learning_rate=Param(initial=0.001, interval=[-4, -2], type='real', scale='log10'),
    learning_rate_decay=Param(initial=0.05, interval=[0, 0.1], type='real'),
    learning_rate_decay_method=Param(initial='sqrt', interval=['exp', 'none', 'sqrt', 'lin'], type='choice'),
    momentum=Param(initial=0.9, interval=[0.5, 0.99], type='real'),
    #weight_decay=Param(initial=0, interval=[-10, -6], type='real', scale='log10'),
    weight_decay=make_constant_param(0),
    max_epochs=make_constant_param(10000),
    batch_size=Param(initial=32,
                     interval=[16, 32, 64, 128],
                     type='choice'),
    patience_nb_epochs=make_constant_param(100),
    patience_threshold=make_constant_param(1),
    patience_check_each=make_constant_param(1),

    optimization=Param(initial='adam',
                       interval=['adam', 'nesterov_momentum', 'rmsprop'],
                       type='choice'),
)


@task
def train(filename=None,
          out=None,
          what="skipthoughts",
          default_model=False,
          budget_hours=np.inf,
          preprocess_whole=1,
          maxlen=None):

    from helpers import get_tokens_from_python_code, get_tokens

    budget_sec = budget_hours * 3600
    print("budget sec : {}".format(budget_sec))

    light = Light()
    light.launch()
    light.initials()
    seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    light.file_snapshot()
    light.set_seed(seed)

    if filename is not None:
        fd = open(filename, "r")
        testing = False
    else:
        filename = "simple.py"
        fd = open(filename, "r")
        testing = True

    light.set("filename", filename)

    if filename.endswith(".py"):
        lines = list(get_tokens_from_python_code(fd))
    else:
        lines = get_tokens(fd)
    fd.close()
    lines = filter(lambda s: len(s) > 0, lines)
    print(len(lines))
    if filename == "simple.py":
        print(lines)
    text = [c for l in lines for c in l]
    chars = list(set(text))
    char_to_int = {c: i + 2 for i, c in enumerate(chars)}
    int_to_char = {i + 2: c for i, c in enumerate(chars)}

    char_to_int[0] = 0
    int_to_char[0] = ""
    char_to_int[""] = 0

    char_to_int[1] = 1
    int_to_char[1] = ""
    char_to_int[""] = 1

    def word2int(w):
        return char_to_int[w]

    def int2word(i):
        return int_to_char[i]

    if maxlen is None:
        maxlen = max(map(len, lines)) + 2
    else:
        maxlen = int(maxlen) + 2
    print("MAXLEN:{}".format(maxlen))
    corpus = lines
    for c in corpus:
        if len(c) == maxlen - 2:
            print(c)
            break

    words = set()
    for sentence in corpus:
        words |= set(sentence)

    nb_words = len(words) + 1
    print("nb  words {} ".format(nb_words))

    def preprocess(corpus, pad=False, left_limit=True,
                   right_limit=True, is_binarize=True):
        corpus = [map(word2int, sentence) for sentence in corpus]
        if pad:
            corpus = pad_sequences_(corpus, maxlen=maxlen - 2)
        if left_limit:
            corpus = map(lambda s: ([1] + s), corpus)
        if right_limit:
            corpus = map(lambda s: (s + [0]), corpus)
        if is_binarize:
            corpus = binarize(corpus)
        return corpus

    def binarize(corpus):
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
        corpus = map(strip, corpus)
        corpus = [map(int2word, sentence) for sentence in corpus]
        return corpus

    def strip(sentence):
        try:
            idx = sentence.index(0)
        except ValueError:
            return sentence
        else:
            sentence = sentence[0:idx]
            return sentence

    def show(corpus, fd=sys.stdout):
        for sentence in corpus:
            sentence = map(lambda c: "0" if c == 0 else c, sentence)
            print(" ".join(sentence), file=fd)
    if preprocess_whole:
        corpus = preprocess(corpus, pad=True)
    prev, cur, next_ = corpus[0:-2], corpus[1:-1], corpus[2:]
    print(len(cur))

    if default_model:
        instantiate = instantiate_default
    else:
        instantiate = instantiate_random

    default_params = {}
    if testing:
        default_params["hidden"] = 64
        instantiate = instantiate_default

    if what == "skipthoughts":
        from pprint import pprint
        light.tag("skipthoughts")
        hp_model = instantiate(skipthoughts_params,
                               default_params=default_params)
        hp = instantiate(params)
        pprint(hp_model)
        pprint(hp)

        if out is not None:
            if os.path.exists(out):
                with open(out, "w") as fd:
                    pass
        train_skipthoughts_(
            maxlen, nb_words, cur, prev, next_,
            preprocess, deprocess, show,
            gen_output=out,
            hp_model=hp_model,
            hp=hp,
            budget_sec=budget_sec,
            preprocess_whole=preprocess_whole)

    elif what == "text_generation":
        if out is not None:
            if os.path.exists(out):
                with open(out, "w") as fd:
                    pass
        hp_model = instantiate(params)
        train_text_generation_(
            maxlen, nb_words, cur, prev, next_,
            preprocess, deprocess, show,
            gen_output=out,
            hp_model=hp_model,
            hp=hp,
            preprocess_whole=preprocess_whole)
    light.endings()  # save the duration

    if testing is False:
        light.store_experiment()  # update the DB
    light.close()


def train_skipthoughts_(maxlen, nb_words, cur, prev, next_,
                        preprocess, deprocess, show,
                        hp_model,
                        hp,
                        gen_output="gen",
                        budget_sec=np.inf,
                        preprocess_whole=True):
    from lasagne import updates, layers
    from lasagnekit.updates import santa_sss
    updates.santa_sss = santa_sss
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.easy import (
        make_batch_optimizer, InputOutputMapping, exp_moving_avg,
        build_batch_iterator,
        get_stat)
    import theano.tensor as T
    from collections import OrderedDict
    import theano
    from StringIO import StringIO
    from lightexperiments.light import Light
    from datetime import datetime

    begin = datetime.now()

    light = Light()
    light.set("hp", hp)
    light.set("hp_model", hp_model)

    build_model = build_model_skipthoughts
    light.set("model", build_model.__name__)
    model_layers = build_model(input_length=maxlen,
                               prev_length=maxlen - 1,
                               next_length=maxlen - 1,
                               nb_words=nb_words,
                               mask=True,
                               **hp_model)

    def categorical_crossentropy_(p, y):
        p = p.reshape((p.shape[0] * p.shape[1], p.shape[2]))
        y = y.reshape((y.shape[0] * y.shape[1],))
        l = -T.log(p[T.arange(p.shape[0]), y])
        return l

    def loss_function(model, tensors):
        input_ = tensors["input"]
        prev = tensors["prev"]
        next_ = tensors["next"]
        prev_output = tensors["prev_output"]
        next_output = tensors["next_output"]
        pred_prev, pred_next = model.get_output(input_, prev, next_)
        return objective(prev, next_,
                         pred_prev,
                         pred_next,
                         prev_output, next_output)

    def objective(prev, next_, pred_prev, pred_next, prev_output, next_output):
        # Prev loss construction
        # mask is 1 if both input and output are character zero
        # prev_mask =  (1 -  (input==character zero) and (output==character zero))
        # which also means :
        # prev_mask = (input != character_zero) or (output != character zero) )
        prev_mask = (1 - T.eq(prev_output.argmax(axis=2, keepdims=True), 0) *
                     T.eq(prev.argmax(axis=2, keepdims=True), 0))
        prev_mask = prev_mask.flatten()
        a, b = pred_prev.shape[0:2]
        loss_prev = categorical_crossentropy_(
            pred_prev, prev_output.argmax(axis=2))
        loss_prev = loss_prev * prev_mask

        # Next loss construction
        next_mask = (1 - T.eq(next_output.argmax(axis=2, keepdims=True), 0) *
                     T.eq(next_.argmax(axis=2, keepdims=True), 0))
        next_mask = next_mask.flatten()
        loss_next = categorical_crossentropy_(
            pred_next, next_output.argmax(axis=2))
        loss_next = loss_next * next_mask
        L = (loss_prev + loss_next)
        L = L.reshape((a, b))
        return L.sum(axis=1).mean()

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

    def shown(G):
        l = []
        for g in G:
            fd = StringIO()
            show(g, fd=fd)
            l.append(fd.getvalue())
        return l

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

        for k, v in status.items():
            light.append(k, float(v))

        #if t % 10 == 0:
        #    report_(status)

        if (datetime.now() - begin).total_seconds() >= budget_sec:
            print("Budget finished.quit.")
            raise KeyboardInterrupt()

        return status

    def report_(status):

        for way in ('argmax', 'proba'):
            for temp in (0.1, 0.5, 1, 2):
                report_gen(status, way=way, temperature=temp)

        # learning curve
        stats = capsule.batch_optimizer.stats
        epoch = get_stat("epoch", stats)
        avg_loss = get_stat("avg_loss_train", stats)
        fig = plt.figure()
        plt.plot(epoch, avg_loss)
        plt.xlabel("x")
        plt.ylabel("avg_loss_train")
        plt.savefig("out/avg_loss_train.png")
        plt.close(fig)

    def report_gen(status, way, temperature=1):
        print("Generating (way={})-------------".format(way))
        gen_init = ['import']
        gen = gen_init
        G = []
        for i in range(40):
            gen = generate_sentence(capsule.get_code, capsule.get_pre_next,
                                    preprocess, deprocess,
                                    gen,
                                    max_length=maxlen,
                                    rng=np.random,
                                    way=way)
            G.append(gen)

        def show_it(fd):
            print("---- epoch {}, way={},temp={}\n".format(status["epoch"], way, temperature), file=fd)
            print("raw generation:\n", file=fd)
            for g in G:
                print(g, file=fd)
            print("real generation:\n", file=fd)
            for g in G:
                show(g, fd=fd)
            print("Finish -----", file=fd)

        if gen_output is not None:
            with open(gen_output, "a") as fd:
                show_it(fd)
        show_it(sys.stdout)

        raw = {
            "temperature": temperature,
            "way": way,
            "content": G,
            "epoch": status["epoch"]
        }
        light.append("rawgen", raw)

        gen = {
            "temperature": temperature,
            "way": way,
            "content": shown(G),
            "epoch": status["epoch"]
        }
        light.append("gen", gen)
        print(gen)

    lr_decay_method = hp["learning_rate_decay_method"]
    initial_lr = hp["learning_rate"]
    lr_decay = hp["learning_rate_decay"]
    lr = theano.shared(np.array(initial_lr, dtype=np.float32))
    optim_params = {"learning_rate": lr}
    if "momentum" in hp["optimization"]:
        optim_params["momentum"] = hp["momentum"]
    optim = (getattr(updates, hp["optimization"]),
             optim_params)
    def transform(batch_index, batch_slice, tensors):
        t = OrderedDict()
        t["input"] = preprocess(tensors["input"][batch_slice], pad=True)
        t["prev"] = preprocess(tensors["prev"][batch_slice], pad=True)
        t["next"] = preprocess(tensors["next"][batch_slice], pad=True)
        t["next_output"] = preprocess(tensors["next_output"][batch_slice], pad=True)
        t["prev_output"] = preprocess(tensors["prev_output"][batch_slice], pad=True)
        capsule.processed += 1
        if capsule.processed % 100 == 0:
            status = {"epoch": capsule.processed}
            report_(status)
        return t

    if preprocess_whole:
        batch_iterator = None
    else:
        batch_iterator = build_batch_iterator(transform)

    batch_optimizer = make_batch_optimizer(
        update_status,
        verbose=1,
        max_nb_epochs=hp["max_epochs"],
        batch_size=hp["batch_size"],
        optimization_procedure=optim,
        patience_stat="avg_loss_train_fix",
        patience_nb_epochs=hp["patience_nb_epochs"],
        patience_progression_rate_threshold=hp["patience_threshold"],
        patience_check_each=hp["patience_check_each"],
        whole_dataset_in_device=False,
    )

    inputs = [model_layers["input"],
              model_layers["prev"],
              model_layers["next"]]
    outputs = [model_layers["prev_output"], model_layers["next_output"]]
    model = InputOutputMapping(inputs, outputs)
    capsule = Capsule(variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer,
                      batch_iterator=batch_iterator)
    capsule.processed = 0

    t_code = T.matrix()
    t_next = capsule.v_tensors["next"]
    v = {
        model_layers["code"]: t_code,
        model_layers["next"]: t_next
    }
    capsule.get_next = theano.function(
        [t_code, t_next],
        layers.get_output(model_layers["next_output"], v))

    capsule.get_pre_next = theano.function(
        [t_code, t_next],
        layers.get_output(model_layers["pre_softmax_next_output"], v))

    # creating dataset
    prev_in = [p[0:-1] for p in prev]
    prev_out = [p[1:] for p in prev]
    next_in = [n[0:-1] for n in next_]
    next_out = [n[1:] for n in next_]
    """
    show(deprocess(cur))
    print("")
    show(deprocess(prev_in))
    print("")
    show(deprocess(next_in))
    print("")
    show(deprocess(prev_out))
    print("")
    show(deprocess(next_out))
    """
    print("nb of examples : {}".format(len(cur)))
    t = OrderedDict({
        "input": cur,
        "prev": prev_in,
        "next": next_in,
        "prev_output": prev_out,
        "next_output": next_out,
    })
    t = OrderedDict(t)
    capsule._build(t)

    try:
        capsule.fit(**t)
    except KeyboardInterrupt:
        print("Interrupted.")


def train_text_generation_(maxlen, nb_words, cur, prev, next_,
                           preprocess, deprocess, show,
                           hidden=512,
                           gen_output="gen"):
    from lasagne import updates, layers
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.easy import (
        make_batch_optimizer, InputOutputMapping, exp_moving_avg,
        get_stat)
    import theano.tensor as T
    from collections import OrderedDict
    import theano
    model_layers = build_model_text_generation(input_length=maxlen,
                                               nb_words=nb_words,
                                               hidden=hidden,
                                               mask=True)

    def categorical_crossentropy_(p, y):
        p = p.reshape((p.shape[0] * p.shape[1], p.shape[2]))
        y = y.reshape((y.shape[0] * y.shape[1],))
        l = -T.log(p[T.arange(p.shape[0]), y])
        return l

    def loss_function(model, tensors):
        input_ = tensors["input"]
        output = tensors["output"]
        pred_output, = model.get_output(input_)
        return objective(input_, pred_output, output)

    def objective(input_, pred_output, output):
        output_mask = (1 - T.eq(output.argmax(axis=2, keepdims=True), 0) *
                       T.eq(input_.argmax(axis=2, keepdims=True), 0))
        output_mask = output_mask.flatten()
        a, b = pred_output.shape[0:2]
        loss = categorical_crossentropy_(
            pred_output, output.argmax(axis=2))
        loss = loss * output_mask
        loss = loss.reshape((a, b))
        return loss.sum(axis=1).mean()

    variables = OrderedDict({
        "input": dict(tensor_type=T.tensor3),
        "output": dict(tensor_type=T.tensor3),
    })

    def get_code(model, input_):
        return layers.get_output(model_layers["code"], input_)

    functions = dict(
        get_code=dict(get_output=get_code, params=["input"]),
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

        for way in ('argmax', 'proba'):
            for temp in (0.1, 0.5, 1, 2):
                report_gen(status, way=way, temperature=temp)

        # learning curve
        stats = capsule.batch_optimizer.stats
        epoch = get_stat("epoch", stats)
        avg_loss = get_stat("avg_loss_train", stats)
        fig = plt.figure()
        plt.plot(epoch, avg_loss)
        plt.xlabel("x")
        plt.ylabel("avg_loss_train")
        plt.savefig("out/avg_loss_train.png")
        plt.close(fig)

    def report_gen(status, way, temperature=1):
        gen_init = [""]
        G = []
        for i in range(10):
            gen = generate_text(capsule.get_pre_next,
                                preprocess, deprocess,
                                gen_init,
                                max_length=30,
                                rng=np.random,
                                way=way)
            G.append(gen)

        def show_it(fd):
            print("---- epoch {}, way={},temp={}\n".format(status["epoch"], way, temperature), file=fd)
            print("raw generation:\n", file=fd)
            for g in G:
                print(g, file=fd)
            print("real generation:\n", file=fd)
            for g in G:
                show(g, fd=fd)
            print("Finish -----", file=fd)

        with open(gen_output, "a") as fd:
            show_it(fd)
        show_it(sys.stdout)


    lr_decay_method = "exp"
    initial_lr = 0.0001
    lr_decay = 0.0000001
    lr = theano.shared(np.array(initial_lr, dtype=np.float32))
    momentum = 0.9
    algo = updates.nesterov_momentum
    params = {"learning_rate": lr}

    if algo == updates.adam:
        params["beta1"] = 0.9
        params["beta2"] = 0.9999

    if algo in (updates.momentum, updates.nesterov_momentum):
        params["momentum"] = momentum
    optim = (algo, params)
    batch_optimizer = make_batch_optimizer(
        update_status,
        max_nb_epochs=100000,
        optimization_procedure=optim,
        patience_stat='avg_loss_train_fix',
        patience_nb_epochs=400,
        min_nb_epochs=100,
        batch_size=128,
        verbose=1)

    inputs = [model_layers["input"]]
    outputs = [model_layers["output"]]
    model = InputOutputMapping(inputs, outputs)
    capsule = Capsule(variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer)

    t_input = T.tensor3()
    v = {
        model_layers["input"]: t_input,
    }
    capsule.get_next = theano.function(
        [t_input],
        layers.get_output(model_layers["output"], v))

    capsule.get_pre_next = theano.function(
        [t_input],
        layers.get_output(model_layers["pre_output"], v))

    # creating dataset
    input_ = [p[0:-1] for p in prev]
    output = [p[1:] for p in prev]
    print(deprocess(input_))
    print("")
    print(deprocess(output))
    show(deprocess(input_))
    t = OrderedDict({
        "input": input_,
        "output": output,
    })
    t = OrderedDict(t)
    capsule._build(t)
    capsule.fit(**t)


def generate_sentence(get_code, get_next,
                      preprocess, deprocess,
                      cur,
                      max_length=10,
                      rng=np.random,
                      way='argmax',
                      temperature=1):
    cur_ = preprocess(cur, pad=True)
    cur_ = np.array(cur_).astype(np.float32)
    nb_words = cur_.shape[2]
    code = get_code(cur_)
    gen = np.zeros((len(cur_), 1, nb_words)).astype(np.float32)
    gen[:, :, 1] = 1  # initialize by the "begin" character

    for i in range(max_length):
        pre_probas = get_next(code, gen)
        probas = softmax(pre_probas * temperature)
        probas = probas[:, -1, :]
        next_gen = []
        for proba in probas:
            if way == 'argmax':
                word_idx = proba.argmax()  # only take argmax
            elif way == 'proba':
                try:
                    proba[-1] = 1 - proba[0:-1].sum()
                    word_idx = rng.multinomial(1, proba).argmax()
                except ValueError:
                    word_idx = proba.argmax()
            else:
                raise Exception("Wrong way : {}".format(way))
            one_hot = [0] * nb_words
            one_hot[word_idx] = 1.
            next_gen.append(one_hot)
        next_gen = np.array(next_gen, dtype="float32")
        next_gen = next_gen[:, None, :]
        gen = np.concatenate((gen, next_gen), axis=1)
    return deprocess(gen)


def generate_text(get_next,
                  preprocess, deprocess,
                  cur,
                  max_length=10,
                  rng=np.random,
                  way='argmax',
                  temperature=1):
    cur_ = preprocess(cur, pad=False, left_limit=True, right_limit=False)
    cur_ = np.array(cur_).astype(np.float32)
    nb_words = cur_.shape[2]
    gen = cur_
    gen = gen.astype(np.float32)

    for i in range(max_length):
        pre_probas = get_next(gen)
        probas = softmax(pre_probas * temperature)
        probas = probas[:, -1, :]
        next_gen = []
        for proba in probas:
            if way == 'argmax':
                word_idx = proba.argmax()  # only take argmax
            elif way == 'proba':
                try:
                    proba[-1] = 1 - proba[0:-1].sum()
                    word_idx = rng.multinomial(1, proba).argmax()
                except ValueError:
                    word_idx = proba.argmax()
            else:
                raise Exception("Wrong way : {}".format(way))
            one_hot = [0] * nb_words
            one_hot[word_idx] = 1.
            next_gen.append(one_hot)
        next_gen = np.array(next_gen, dtype="float32")
        next_gen = next_gen[:, None, :]
        gen = np.concatenate((gen, next_gen), axis=1)
    return deprocess(gen)
