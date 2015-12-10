from invoke import task


@task
def train(word2vec_filename="data/glove.6B.300d.pkl"):
    from model import build_visual_model
    from caffezoo.googlenet import GoogleNet
    from lasagne import layers
    import theano.tensor as T
    from helpers import load_word_embedding

    # visual model
    googlenet = GoogleNet()
    googlenet._load()
    base = googlenet.net
    model = build_visual_model(base,
                               input_layer_name="input",
                               repr_layer_name="pool1/norm1")
    input_layer, output_layer = model
    # word model
    word2vec = load_word_embedding(word2vec_filename)
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
