import pickle


def load_word_embedding(filename):
    with open(filename, "r") as fd:
        data = pickle.load(fd)
    return data
