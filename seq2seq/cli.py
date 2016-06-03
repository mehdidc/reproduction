import click
import time

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
@click.option('--net', default='conv', help='which formulation of the problem to use : rnn_next_char/rnn_next_chars/rnn_skipthought/rnn_aa/conv_aa/dense_aa', required=False)
@click.option('--data', default='harry.txt', help='which data to use', required=False)
@click.option('--split', default='sliding_window', help='sliding_window/jumping_window', required=False)
@click.option('--granularity', default='char', help='char/word', required=False)
@click.option('--seq_length', default=50, help='sequence length', required=False)
@click.option('--hidden', default=256, help='hidden units number', required=False)
@click.option('--seq_model', default='rnn', help='rnn/gru/lstm', required=False)
@click.option('--nb_layers', default=1, help='nb_layers', required=False)
@click.option('--batchsize', default=128, help='batch size', required=False)
@click.option('--out', default='out', help='outfile', required=False)
@click.option('--dropout', default=0., help='dropout probability', required=False)
@click.option('--stateful', default=False, help='True/False', required=False)
def train(net, data, split, granularity, seq_length, hidden, seq_model, nb_layers, batchsize, out, dropout, stateful):
    T = seq_length # max seq length
    H = hidden  # hidden layer size
    E = hidden # embedding size
    mode = net
    text = open(data).read()
    assert granularity in ('char', 'word')
    if granularity == 'char':
        pass
    elif granularity == 'word':
        sent = get_tokens(text)
        sent  = map(get_subtokens, sent)
        text  = [w for s in sent for w in s]
    T_ = T + 1 if mode == 'rnn_next_chars' else T
    assert split in ('sliding_window', 'jumping_window')
    if split == 'sliding_window':
        sent = text
        sent = [sent[i: i + T_] for i in range(0, len(sent))]
    elif split == 'jumping_window':
        sent = text
        sent = [sent[i: i + T_] for i in range(0, len(sent), T_)]
    v = DocumentVectorizer(length=T_, pad=True, begin_letter=False)
    X = v.fit_transform(sent)
    X = intX(X)
    D = len(v._word2int)
    print(D)
    SeqModel = {'rnn': SimpleRNN, 'gru': GRU, 'lstm': LSTM}[seq_model]

    if mode == 'rnn_next_char':
        # like Keras and Lasagne examples
        xcur = Input(batch_shape=(batchsize, T), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xcur)
        h = x
        for i in range(nb_layers):
            rs = False if i == nb_layers - 1 else True
            h = SeqModel(H, stateful=stateful, init='orthogonal', return_sequences=rs)(h)
            if dropout > 0:
                h = Dropout(dropout)(h)
        x = Dense(D)(h)
        pre_model = Model(input=xcur, output=x)
        x = Activation('softmax')(x)
        outp = x
        model = Model(input=xcur, output=outp)
    if mode == 'rnn_next_chars':
        # like Karpathy's char-rnn
        xcur = Input(batch_shape=(batchsize, T), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xcur)
        for i in range(nb_layers):
            h = SeqModel(H, stateful=stateful, init='orthogonal', return_sequences=True)(x)
            if dropout > 0:
                h = Dropout(dropout)(h)

        x = TimeDistributed(Dense(D))(h)
        pre_model = Model(input=xcur, output=x)
        x = Activation('softmax')(x)
        outp = x
        model = Model(input=xcur, output=outp)
    if mode == 'rnn_skipthought':
        # like skipthought ? to check
        # encoder
        xcur = Input(batch_shape=(batchsize, T), dtype='int32')
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(xcur)
        h = SeqModel(H, init='orthogonal', return_sequences=True, stateful=False)(x)
        h = RepeatVector(T)(h)
        # decoder
        xnext = Input(shape=(T,), dtype='int32')
        x = SeqModel(input_dim=D, output_dim=E, input_length=T)(xnext)
        x = merge([h, x], mode='concat')
        x = SeqModel(D, init='orthogonal', return_sequences=True, stateful=False)(x)
        x = Activation('softmax')(x)
        outp = x
        model = Model(input=[xcur, xnext], output=outp)
    elif mode == 'rnn_aa':
        # like Sequence to Sequence Learning with Neural Networks
        xcur = Input(shape=(T,), dtype='int32', name='cur')
        x = xcur
        x = Embedding(input_dim=D, output_dim=E, input_length=T)(x)
        h = SeqModel(H, init='orthogonal')(x)
        h = RepeatVector(T)(h)
        h = SeqModel(H, init='orthogonal', return_sequences=True)(h)
        x = TimeDistributed(Dense(D))(h)
        x = Activation('softmax', name='next')(x)
        xnext  = x
        model = Model(input=xcur, output=xnext)
    elif mode == 'conv_aa':
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
    elif mode == 'dense_aa':
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

    optimizer = RMSprop(lr=0.0001, #(0.0001 for rnn_next_char, 0.001 for conv_aa)
                        #clipvalue=5,
                        rho=0.95,
                        epsilon=1e-8)
    def clean(s):
        return map(lambda w:'' if w in (0, 1) else w, s)

    def show(gen, files):
        for g in gen:
            for fd in files:
                g = clean(g)
                s = ''.join(g)
                fd.write('\n')
                fd.write(s)
                fd.write('\n')
                fd.write('-'*min(80, len(s)))
                fd.write('\n')

    def gen(epoch):
        if mode == 'rnn_next_char':
            def pred_func(gen):
                return pre_model.predict_on_batch([gen])
            i = np.random.randint(0, len(inp[0]))
            cur = np.repeat(inp[0][i:i+1], batchsize, axis=0)
            gen = generate_text(pred_func, v, cur=cur, nb=batchsize if stateful else 10, max_length=T * 20, rng=np.random, way='proba', temperature=1)
            fd = open(out, 'a')
            show(gen, [sys.stdout, fd])
            fd.close()
        if mode == 'rnn_next_chars':
            def pred_func(gen):
                return pre_model.predict_on_batch([gen])[:, -1, :]
            i = np.random.randint(0, len(inp[0]))
            cur = np.repeat(inp[0][i:i+1], batchsize, axis=0)
            gen = generate_text(pred_func, v, cur=cur, nb=batchsize if stateful else 10, max_length=T * 20, rng=np.random, way='proba', temperature=1)
            fd = open(out, 'a')
            show(gen, [sys.stdout, fd])
            fd.close()
        if mode == 'rnn_skipthought':
            pass
        if mode in ('rnn_aa', 'conv_aa', 'dense_aa'):
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

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    if mode == 'rnn_skipthought':
        raise NotImplementedError()
    if mode == 'rnn_next_chars':
        inp = [X[:, 0:-1]]
        outp = X[:, 1:]
        pr_noise = 0
    if mode == 'rnn_next_char':
        inp = [X[0:-1, :]]
        outp = X[1:, -1][:, None]
        pr_noise = 0
    if mode == 'rnn_aa':
        inp = [X[0:-1, :]]
        outp = inp[0] # reverse oof the input like sequence to sequence learning with neural networks paper
        pr_noise = 0
    if mode in ('conv_aa', 'dense_aa'):
        inp = [X[0:-1, :]]
        outp = inp[0]
        pr_noise = 0.5
    avg_loss = -np.log(1./D)
    nb = 0
    for i in range(10000):
        t = time.time()
        for s in iterate_minibatches(len(outp), batchsize=batchsize, exact=True):
            x_mb_orig = [x[s] for x in inp]
            x_mb = [noise(x[s], pr=pr_noise) for x in inp]
            y_mb = categ(outp[s], D=D)
            if y_mb.shape[1] == 1:
                y_mb = y_mb[:, 0, :]
            model.fit(x_mb, y_mb, nb_epoch=1, batch_size=batchsize, verbose=0)
            loss = model.evaluate(x_mb_orig, y_mb, verbose=0, batch_size=batchsize)
            avg_loss = avg_loss * 0.999 + loss * 0.001
            nb += 1
            if nb % 100 == 0 and stateful is False:
                gen(i)
        print('Full data pass time in sec : {}'.format(time.time() - t))
        print('avg loss : {}, nb updates : {}'.format(avg_loss, nb))
        if stateful:
            model.reset_states()
            gen(i)
            model.reset_states()

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
                word_idx = np.random.choice(np.arange(len(word_pr)), p=word_pr)
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
