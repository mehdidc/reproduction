from invoke import task
import pickle


@task
def word_embedding_to_binary(filename, out_filename):
    words = dict()
    with open(filename) as fd:
        for line in fd.readlines():
            components = line.split(" ")
            word = components[0]
            embedding = map(float, components[1:])
            words[word] = embedding
    with open(out_filename, "w") as fd:
        pickle.dump(words, fd)
