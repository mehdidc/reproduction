import click
from keras.preprocessing import sequence
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.callbacks import LearningRateScheduler, Callback
from keras.utils.np_utils import to_categorical
from scipy.stats import entropy

class DocumentVectorizer(object):

    def __init__(self, length=None, begin_letter=True, pad=True):
        self.length = length
        self.begin_letter = begin_letter
        self.pad = pad

    def fit(self, docs):
        all_words = set(word for doc in docs for word in doc)
        all_words = set(all_words)
        all_words.add(0)
        all_words.add(1)
        self._nb_words = len(all_words)
        self._word2int = {w: i for i, w in enumerate(all_words)}
        self._int2word = {i: w for i, w in enumerate(all_words)}
        return self

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def _doc_transform(self, doc):
        doc = map(self._word_transform, doc)
        if self.length:
            if len(doc) >= self.length:
                return doc[0:self.length]
            else:
                doc_new = []
                if self.begin_letter:
                    doc_new.append(self._word_transform(1))
                doc_new.extend(doc)
                if self.pad:
                    remaining = self.length - len(doc_new)
                    doc_new.extend(map(self._word_transform, [0] * remaining))
                return doc_new
        else:
            return doc

    def _word_transform(self, word):
        return self._word2int[word]

    def transform(self, docs):
       docs = map(self._doc_transform, docs)
       if self.length:
           docs = np.array(docs)
       return docs

    def inverse_transform(self, X):
        docs = []
        for s in X:
            docs.append([self._int2word[w] for w in s])
        return docs

class LambdaScheduler(Callback):
    def __init__(self, schedule):
        super(LambdaScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs={}):
        self.schedule(epoch)

@click.group()
def main():
    pass

@click.command()
@click.option('--net', default='conv', help='which network to use : lstm/lstm-aa/conv', required=False)
@click.option('--data', default='harry.txt', help='which data to use', required=False)
@click.option('--split', default='sliding_window', help='sliding_window', required=False)
@click.option('--granularity', default='char', help='char/word', required=False)
@click.option('--seq_length', default=50, help='seq_length', required=False)
@click.option('--hidden', default=256, help='hidden units number', required=False)
def train(net, data, split, granularity, seq_length, hidden):

    T = seq_length # max seq length
    H = hidden  # hidden layer size
    E = hidden # embedding size
    batchsize = 128
    mode = net
    text = open(data).read()
    if granularity == 'char':
        pass
    elif granularity == 'word':
        sent = get_tokens(text)
        sent  = map(get_subtokens, sent)
        text  = [w for s in sent for w in s]

    if split == 'sliding_window':
        sent = text
        sent = [sent[i: i + T] for i in range(0, len(sent))]
    elif split == 'jumping_window':
        sent = text
        sent = [sent[i: i + T] for i in range(0, len(sent), T)]
    #sent = sent[0:256]

    v = DocumentVectorizer(length=T, pad=True, begin_letter=False)
    X = v.fit_transform(sent)
    X = intX(X)
    D = len(v._word2int)
    if mode == 'lstm_simple':
        xcur = Input(batch_shape=(batchsize, T), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xcur)
        h = SimpleRNN(H, stateful=True, init='orthogonal', return_sequences=True)(x)
        h = SimpleRNN(H, init='orthogonal')(x)
        x = Dense(D)(h)
        pre_model = Model(input=xcur, output=x)
        x = Activation('softmax')(x)
        outp = x
        model = Model(input=xcur, output=outp)
    if mode == 'lstm':
        # encoder
        xcur = Input(batch_shape=(batchsize, T), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xcur)
        h = SimpleRNN(H, init='orthogonal', return_sequences=True, stateful=True)(x)
        h = RepeatVector(T)(h)
        # decoder
        xnext = Input(shape=(T,), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xnext)
        x = merge([h, x], mode='concat')
        x = SimpleRNN(D, init='orthogonal', return_sequences=True, stateful=True)(x)
        x = Activation('softmax')(x)
        outp = x
        model = Model(input=[xcur, xnext], output=outp)
    elif mode == 'lstm-aa':
        # like Sequence to Sequence Learning with Neural Networks
        xcur = Input(shape=(T,), dtype='int32', name='cur')
        x = xcur
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(x)
        h = SimpleRNN(H, init='orthogonal')(x)
        h = RepeatVector(T)(h)
        h = SimpleRNN(H, init='orthogonal', return_sequences=True)(h)
        x = TimeDistributed(Dense(D))(h)
        x = Activation('softmax', name='next')(x)
        xnext  = x
        model = Model(input=xcur, output=xnext)
    elif mode == 'conv-aa':
        x = Input(shape=(T,), dtype='int32', name='cur')
        xcur = x
        x = Embedding(input_dim=D, output_dim=D, input_length=T)(x)
        x = Convolution1D(64, 10)(x)
        x = Activation('relu')(x)
        x = Convolution1D(128, 10)(x)
        x = Activation('relu')(x)

        x = ZeroPadding1D(padding=10 - 1)(x)
        x = Convolution1D(256, 10)(x)
        x = Activation('relu')(x)

        x = ZeroPadding1D(padding=10 - 1)(x)
        x = Convolution1D(D, 10)(x)

        x = Activation('softmax')(x)
        xnext = x
        model = Model(input=xcur, output=xnext)
    elif mode == 'dense-aa':
        x = Input(shape=(T,), dtype='int32', name='cur')
        xcur = x
        x = Embedding(input_dim=D, output_dim=D, input_length=T)(x)
        x = Reshape((T*D,))(x)
        x = Dense(H)(x)
        x = Activation('relu')(x)
        x = Dense(T*D)(x)
        x = Reshape((T, D))(x)
        x = Activation('softmax')(x)
        xnext = x
        model = Model(input=xcur, output=xnext)

    optimizer = RMSprop(lr=0.0001, #(0.0001 for lstm_simple, 0.001 for conv-aa)
                        #clipvalue=5,
                        rho=0.95,
                        epsilon=1e-8)
    def scheduler(epoch):
        lr = model.optimizer.lr.get_value() * 0.999
        lr = float(lr)
        model.optimizer.lr.set_value(lr)
        return lr

    change_lr = LearningRateScheduler(scheduler)

    def clean(s):
        return map(lambda w:'' if w in (0, 1) else w, s)

    def gen(epoch):
        if mode == 'lstm_simple':
            # then generate
            def pred_func(gen):
                return model.predict([gen])
            i = np.random.randint(0, len(inp[0]))
            cur = np.repeat(inp[0][i:i+1], batchsize, axis=0)
            gen = generate_text(pred_func, v, cur=cur, nb=batchsize, max_length=T * 4, rng=np.random, way='proba', temperature=2)
            for g in gen[0:10]:
                g = clean(g)
                s = ''.join(g)
                print('\n')
                print(s)
                print('-' * len(s))
                print('\n')
        if mode == 'lstm':
            pass
        if mode in ('lstm-aa', 'conv-aa', 'dense-aa'):
            i = np.random.randint(0, len(X))
            inputs = X[i:i+1]
            preds = model.predict(X[i:i + 1])
            preds = preds.argmax(axis=-1)

            pred_words = v.inverse_transform(preds)
            input_words = v.inverse_transform(inputs)

            real = input_words[0]
            real = clean(real)
            real = ''.join(real)

            pred = pred_words[0]
            pred = clean(pred)
            pred = ''.join(pred)

            print('Epoch {}'.format(epoch))
            print('---------')
            print('real')
            print('----')
            print(real)

            print('pred')
            print('----')
            print(pred)
            nb = 10
            cur = np.random.randint(0, D, size=(nb, T))
            gen = v.inverse_transform(iterative_generation(cur, model))
            print("gen")
            print("---")
            for g in gen:
                g = clean(g)
                print(''.join(g))
                print('*' * len(g))
            print('\n')


    callbacks = [change_lr, LambdaScheduler(gen)]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    if mode == 'lstm':
        raise NotImplementedError()
    if mode == 'lstm_simple':
        inp = [X[0:-1, :]]
        outp = X[1:, -1][:, None]
        pr_noise = 0
    if mode == 'lstm-aa':
        inp = [X[0:-1, :]]
        outp = inp[0] # reverse oof the input like sequence to sequence learning with neural networks paper
        pr_noise = 0
    if mode in ('conv-aa', 'dense-aa'):
        inp = [X[0:-1, :]]
        outp = inp[0]
        pr_noise = 0.5
    avg_loss = 0.
    for i in range(10000):
        nb =0
        for s in iterate_minibatches(len(outp), batchsize=batchsize, exact=True):
            x_mb_orig = [x[s] for x in inp]
            x_mb = [noise(x[s], pr=pr_noise) for x in inp]
            y_mb = categ(outp[s], D=D)
            if y_mb.shape[1] == 1:
                y_mb = y_mb[:, 0, :]
            model.fit(x_mb, y_mb, nb_epoch=1, batch_size=batchsize, verbose=0)#, callbacks=callbacks)
            loss = model.evaluate(x_mb_orig, y_mb, verbose=0, batch_size=batchsize)
            avg_loss = avg_loss * 0.9 + loss * 0.1
            nb += 1
        print('avg loss : {}'.format(avg_loss))
        gen(i)

def categ(X, D=10):
    return np.array([to_categorical(x, nb_classes=D) for x in X])

def noise(x, pr=0.3):
    if pr == 0:
        return x
    m = (np.random.uniform(size=x.shape) <= pr)
    return x * m + np.random.randint(0, x.max(), size=x.shape) * (1 - m)

def get_subtokens(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

def get_tokens(text):
    import nltk.data
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_detector.tokenize(text)

def iterative_generation(cur, model, nb_iter=20):
    for i in range(nb_iter):
        prev = cur
        cur = model.predict(cur)
        score = np.mean(cur.argmax(axis=-1) != prev.argmax(axis=-1))
        cur = cur.argmax(axis=-1)
        #cur = sample(cur)
        if score == 0:
            break
    return cur

def sample(X, temperature=1, rng=np.random):
    s = np.empty((X.shape[0], X.shape[1]))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            pr = softmax(X[i, j] * temperature)
            s[i, j] = rng.multinomial(1, pr).argmax()
    return s


def generate_text(pred_func, vectorizer, cur=None, nb=1, max_length=10, rng=np.random, way='argmax', temperature=1):
    """
    cur        : cur text to condition on (seed), otherwise initialized by the begin character, shape = (N, T)
    pred_func  : function which predicts the next character based on a set of characters, it takes (N, T) as input and returns (N, D) as output
                 where T is the number of time steps, N size of mini-batch and D size of vocabulary
    max_length : nb of characters to generate
    nb : nb of samples to generate (from the same seed)
    """
    assert way in ('proba', 'argmax')
    nb_words = len(vectorizer._word2int)
    if cur is None:
        # initialize the 'seed' with random words
        gen = np.random.randint(0, vectorizer._nb_words,
                                size=(nb, vectorizer.length + max_length))
        start = vectorizer.length
    else:
        gen = np.ones((len(cur), cur.shape[1] + max_length))
        start = cur.shape[1]
        gen[:, 0:start] = cur
    gen = intX(gen)
    for i in range(start, start + max_length):
        pr = pred_func(gen[:, i - start:i])
        pr = softmax(pr * temperature)
        next_gen = []
        for word_pr in pr:
            if way == 'argmax':
                word_idx = word_pr.argmax()  # only take argmax
            elif way == 'proba':
                word_idx = rng.multinomial(1, word_pr).argmax()
            next_gen.append(word_idx)
        gen[:, i] = next_gen
    return vectorizer.inverse_transform(gen[:, start:])

def floatX(x):
    return np.array(x).astype(np.float32)

def intX(x):
    return np.array(x).astype(np.int32)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def iterate_minibatches(nb,  batchsize, exact=False):
    if exact:
        r = range(0, (nb/batchsize) * batchsize, batchsize)
    else:
        r = range(0, nb, batchsize)
    for start_idx in r:
        S = slice(start_idx, start_idx + batchsize)
        yield S


main.add_command(train)

if __name__ == "__main__":
    main()
