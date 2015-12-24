from keras.models import Graph
from keras.layers.recurrent import GRU
from keras.layers.core import TimeDistributedDense, RepeatVector
from lasagne.layers import InputLayer, GRULayer, NonlinearityLayer
from lasagnekit.easy import layers_from_list_to_dict
from lasagne import nonlinearities
from helpers import CondGRULayer, TensorDenseLayer
import theano.tensor as T

def build_model_keras(input_length=None, nb_words=None, hidden=512):
    prev_length = input_length
    next_length = input_length
    graph = Graph()
    graph.add_input(name="input", input_shape=(input_length, nb_words))
    graph.add_node(GRU(output_dim=hidden, return_sequences=False),
                   name="code", input="input")
    graph.add_node(RepeatVector(input_length), input="code", name="code_repeat")
    print(graph.nodes["code_repeat"].output_shape)
    graph.add_node(GRU(output_dim=prev_length, return_sequences=True),
                   input="code_repeat",
                   name="prev")
    graph.add_node(TimeDistributedDense(nb_words, activation="softmax"),
                   name="prev_softmax", input="prev")
    graph.add_output(name="prev_output", input="prev_softmax")
    graph.add_node(GRU(output_dim=next_length, return_sequences=True),
                   input="code_repeat",
                   name="next")
    graph.add_node(TimeDistributedDense(nb_words, activation="softmax"),
                   name="next_softmax", input="next")
    graph.add_output(name="next_output", input="next_softmax")
    return graph


def build_model_lasagne(
        input_length=None, prev_length=None, next_length=None,
        nb_words=None, hidden=512, mask=False):
    l_input = InputLayer((None, input_length, nb_words), name="input")
    l_prev_input = InputLayer((None, prev_length, nb_words), name="prev")
    l_next_input = InputLayer((None, next_length, nb_words), name="next")
    if mask:
        l_input_mask = InputLayer((None, input_length))
        l_prev_mask = InputLayer((None, prev_length))
        l_next_mask = InputLayer((None, next_length))

    l_code = l_input 
    l_code = GRULayer(l_code, num_units=hidden,
                      only_return_final=True, name="code")

    l_pre_prev_output = CondGRULayer(
        l_prev_input,
        num_units=hidden, bias=l_code, 
        name="pre_prev_output")
    #l_prev_output = NonlinearityLayer(
    #    l_pre_prev_output,
    #    softmax_,
    #    name="prev_output"
    #)
    l_prev_output = TensorDenseLayer(
        l_pre_prev_output,
        num_units=nb_words,
        name="prev_output",
        nonlinearity=softmax_
    )

    l_pre_next_output = CondGRULayer(
        l_next_input,
        num_units=hidden, bias=l_code, 
        name="pre_next_output")
    l_next_output = TensorDenseLayer(
        l_pre_next_output,
        num_units=nb_words,
        name="next_output",
        nonlinearity=softmax_
    )
    #l_next_output = NonlinearityLayer(
    #    l_pre_next_output,
    #    softmax_,
    #    name="next_output"
    #)
    layers = [l_input, l_prev_input,
              l_code,
              l_next_input, l_prev_output, l_next_output]
    return layers_from_list_to_dict(layers)

def softmax_(x):
    return T.exp(x)/T.exp(x).sum(axis=-1, keepdims=True)
