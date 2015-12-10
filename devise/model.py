from lasagne.layers import DenseLayer


def build_visual_model(base,
                       input_layer_name="input",
                       repr_layer_name="fc",
                       embedding_size=500):
    l_input = base[input_layer_name]
    l_repr = base[repr_layer_name]
    l_hidden = DenseLayer(l_repr, num_units=embedding_size)
    return l_input, l_hidden