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

    def __init__(self, length=None):
        self.length = length

    def fit(self, docs):
        all_words = set(word for doc in docs for word in doc)
        all_words = set(all_words)
        all_words.add(0)
        all_words.add(1)
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
                return [self._word_transform(1)] + doc + map(self._word_transform, [0] * (self.length - 1 - len(doc)))
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
def train(net, data):
    T = 50 # max seq length
    H = 256  # hidden layer size
    E = 256 # embedding size
    mode = net
    text = open(data).read()
    sent = text
    #sent = get_tokens(text)
    #sent  = map(get_subtokens, sent)

    #sent = [sent[i: i + T] for i in range(0, len(sent), T)]
    sent = [sent[i: i + T] for i in range(0, len(sent))]
    #sent = sent[0:2]
    print(sent[0])

    v = DocumentVectorizer(length=T + 1)
    X = v.fit_transform(sent)
    X = intX(X)
    D = len(v._word2int)

    if mode == 'lstm':
        # encoder
        xcur = Input(shape=(T,), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xcur)
        h = SimpleRNN(H, init='orthogonal')(x)
        h = RepeatVector(T)(h)
        # decoder
        xnext = Input(shape=(T,), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xnext)
        x = merge([h, x], mode='concat')
        x = SimpleRNN(D, init='orthogonal', return_sequences=True)(x)
        x = Activation('softmax')(x)
        outp = x
        model = Model(input=[xcur, xnext], output=outp)
    elif mode == 'lstm-aa':
        xcur = Input(shape=(T,), dtype='int32', name='cur')
        x = xcur
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(x)
        h = LSTM(H, return_sequences=True)(x)
        h = LSTM(H)(h)
        h = RepeatVector(T)(h)
        x = LSTM(D, return_sequences=True)(h)
        x = Activation('softmax', name='next')(x)
        xnext  = x
        model = Model(input=xcur, output=xnext)
    elif mode == 'conv-aa':
        x = Input(shape=(T,), dtype='int32', name='cur')
        xcur = x
        x = Embedding(input_dim=D, output_dim=D, input_length=T)(x)
        x = Convolution1D(64, 10)(x)
        x = Convolution1D(128, 10)(x)

        x = ZeroPadding1D(padding=10 - 1)(x)
        x = Convolution1D(256, 10)(x)

        x = ZeroPadding1D(padding=10 - 1)(x)
        x = Convolution1D(D, 10)(x)

        x = Activation('softmax')(x)
        xnext = x
        model = Model(input=xcur, output=xnext)

    optimizer = RMSprop(lr=0.0001,
                        clipvalue=200,
                        rho=0.95,
                        epsilon=1e-8)
    def scheduler(epoch):
        lr = model.optimizer.lr.get_value() * 0.999
        lr = float(lr)
        model.optimizer.lr.set_value(lr)
        return lr

    change_lr = LearningRateScheduler(scheduler)

    def clean(s):
        return map(lambda w:'' if w in (0, 1) else '\\n' if w in ('\n', '\r') else w, s)

    def gen(epoch):
        if mode == 'lstm':
            for i in range(10):
                j = np.random.randint(0, len(Xcur))
                cur = Xcur[j:j + 1]
                gen = generate_text(cur, model, v, max_length=T, rng=np.random, way='argmax', temperature=2)
                gen = gen[0]
                gen = clean(gen)
                print(''.join(gen))
        if mode == 'lstm-aa' or mode == 'conv-aa':
            i = np.random.randint(0, len(Xcur))
            words = Xcur_words[i:i+1]
            pred = model.predict(Xcur[i:i + 1])
            pred = pred.argmax(axis=-1)
            pred_words = v.inverse_transform(pred)

            real = words[0]
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
            nb = 1
            cur = np.random.randint(0, D, size=(nb, T))
            gen = v.inverse_transform(iterative_generation(cur, model))[0]
            gen = clean(gen)
            print("gen")
            print("---")
            print(''.join(gen))
            print('\n')

    callbacks = [change_lr, LambdaScheduler(gen)]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    Xcur, Xnext, Xnext_shifted = X[0:-1, 0:-1], X[1:, 0:-1], X[1:, 1:]

    Xcur_words = v.inverse_transform(Xcur)
    Xnext_words = v.inverse_transform(Xnext)

    if mode == 'lstm':
        inp = [Xcur, Xnext]
        outp = Xnext_shifted
        pr_noise = 0
    if mode == 'lstm-aa':
        inp = [Xcur]
        outp = Xnext
        pr_noise = 0
    if mode == 'conv-aa':
        inp = [Xcur]
        outp = Xnext
        pr_noise = 0.3
    avg_loss = 0.
    for i in range(10000):
        nb =0
        for s in iterate_minibatches(len(outp), batchsize=128):
            x_mb = [noise(x[s], pr=pr_noise) for x in inp]
            y_mb = categ(outp[s], D=D)
            model.fit(x_mb, y_mb, nb_epoch=1, batch_size=128, verbose=0)#, callbacks=callbacks)
            loss = model.evaluate(x_mb, y_mb, verbose=0)
            avg_loss = avg_loss * 0.9 + loss * 0.1
            nb += 1
        print('avg loss : {}'.format(avg_loss))
        #scheduler(i)
        gen(i)

def categ(X, D=10):
    return np.array([to_categorical(x, nb_classes=D) for x in X])

def noise(x, pr=0.3):
    if pr == 0:
        return x
    return x * (np.random.uniform(size=x.shape) <= pr)

def get_subtokens(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

def get_tokens(text):
    import nltk.data
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_detector.tokenize(text)

def iterative_generation(cur, model, nb_iter=10):
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


def generate_text(cur, model, vectorizer, max_length=10, rng=np.random, way='argmax', temperature=1):
    nb_words = len(vectorizer._word2int)
    gen = np.ones((len(cur), max_length))
    gen = intX(gen)

    for i in range(max_length - 1):
        pre_probas = model.predict([cur, gen])
        probas = softmax(pre_probas * temperature)
        probas = probas[:, i, :]
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
            next_gen.append(word_idx)
        gen[:, i + 1] = next_gen
    return vectorizer.inverse_transform(gen)

def floatX(x):
    return np.array(x).astype(np.float32)

def intX(x):
    return np.array(x).astype(np.int32)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def iterate_minibatches(nb,  batchsize):
    for start_idx in range(0, nb, batchsize):
        S = slice(start_idx, start_idx + batchsize)
        yield S


main.add_command(train)

if __name__ == "__main__":
    main()
