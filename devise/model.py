from lasagne.layers import DenseLayer
from lasagne.nonlinearities import linear


def build_visual_model(base,
                       input_layer_name="input",
                       repr_layer_name="fc",
                       size_embedding=500):
    l_input = base[input_layer_name]
    l_repr = base[repr_layer_name]
    l_hidden = DenseLayer(l_repr, num_units=size_embedding,
                          nonlinearity=linear,
                          name="embed")
    return l_input, l_hidden
