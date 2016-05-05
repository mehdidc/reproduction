from lasagne.layers import InputLayer, GRULayer, ExpressionLayer, NonlinearityLayer
from lasagne.init import Orthogonal
from lasagne.nonlinearities import tanh, sigmoid, linear
from lasagne.layers.recurrent import Gate
from lasagnekit.easy import layers_from_list_to_dict
from helpers import CondGRULayer, TensorDenseLayer
import theano.tensor as T
from hp_toolkit.hp import Param

# steeper sigmoid


skipthoughts_params = dict(
    nb_layers=Param(initial=1, interval=[1, 2], type='int'),
    hidden=Param(initial=512, interval=[64, 800], type='int'),
    grad_clipping=Param(initial=0, interval=[0, 50], type='int'),
    gate_nonlin=Param(initial='sigmoid', interval=['sigmoid'], type='choice'),
    gate_mul=Param(initial=1, interval=[1, 5], type='real')
)


def build_model_skipthoughts(
        input_length=None, prev_length=None, next_length=None,
        nb_words=None,
        mask=True,
        **hp):

    hidden = hp["hidden"]
    nb_layers = hp["nb_layers"]
    grad_clipping = hp["grad_clipping"]

    nonlins = {
        "sigmoid": sigmoid
    }

    def nonlin(x):
        return nonlins[hp["gate_nonlin"]](x * hp["gate_mul"])

    l_input = InputLayer((None, input_length, nb_words), name="input")
    l_prev_input = InputLayer((None, prev_length, nb_words), name="prev")
    l_next_input = InputLayer((None, next_length, nb_words), name="next")
    if mask:
        def construct_mask(X):
            isnt_zero = 1 - (T.eq(X.argmax(axis=2, keepdims=True), 0))
            mask = isnt_zero
            mask = T.set_subtensor(mask[:, 0, 0], 1)
            # mask = 1 if input character isnt the zero character 0 if it is
            # unless it is the first character of the sentence, which is always
            # the zero character, so only the padding zeros (after the end of
            # the sentence) are ignored
            return mask
        l_input_mask = ExpressionLayer(l_input, construct_mask)
    else:
        l_input_mask = None

    def build_gate():
        return Gate(W_cell=None, W_hid=Orthogonal(), nonlinearity=nonlin)

    def build_hid_gate():
        return Gate(W_cell=None, W_hid=Orthogonal(), nonlinearity=tanh)

    l_code = l_input

    # nb layers of GRU to compute the code
    for i in range(nb_layers):
        if i == nb_layers - 1:
            only_final = True
        else:
            only_final = False
        l_code = GRULayer(l_code, num_units=hidden,
                          resetgate=build_gate(),
                          updategate=build_gate(),
                          hidden_update=build_hid_gate(),
                          mask_input=l_input_mask,
                          grad_clipping=grad_clipping,
                          only_return_final=only_final,
                          name="code")
    # having the code, generate prev and next sentences

    # prev
    l_pre_prev_output = CondGRULayer(
        l_prev_input,
        resetgate=build_gate(),
        updategate=build_gate(),
        hidden_update=build_hid_gate(),
        num_units=hidden, bias=l_code,
        grad_clipping=grad_clipping,
        name="pre_prev_output")
    l_pre_softmax_prev_output = TensorDenseLayer(
        l_pre_prev_output,
        num_units=nb_words,
        name="pre_softmax_prev_output",
        nonlinearity=linear,
        b=None,
    )
    l_prev_output = NonlinearityLayer(
        l_pre_softmax_prev_output,
        nonlinearity=softmax_,
        name="prev_output"
    )
    # next
    l_pre_next_output = CondGRULayer(
        l_next_input,
        resetgate=build_gate(),
        updategate=build_gate(),
        hidden_update=build_hid_gate(),
        num_units=hidden,
        bias=l_code,
        grad_clipping=grad_clipping,
        name="pre_next_output")
    l_pre_softmax_next_output = TensorDenseLayer(
        l_pre_next_output,
        num_units=nb_words,
        name="pre_softmax_next_output",
        nonlinearity=linear,
        W=l_pre_softmax_prev_output.W,
        b=None
    )
    l_next_output = NonlinearityLayer(
        l_pre_softmax_next_output,
        nonlinearity=softmax_,
        name="next_output"
    )
    # all layers together
    layers = [l_input,
              l_prev_input,
              l_code,
              l_next_input,
              l_pre_softmax_prev_output,
              l_prev_output,
              l_pre_softmax_next_output,
              l_next_output]
    return layers_from_list_to_dict(layers)


def build_model_text_generation(
        input_length=None,
        nb_words=None,
        mask=True,
        **hp):

    hidden = hp["hidden"]
    nb_layers = hp["nb_layers"]
    grad_clipping = hp["grad_clipping"]

    nonlins = {
        "sigmoid": sigmoid
    }

    def nonlin(x):
        return nonlins[hp["gate_nonlin"]](x * hp["gate_mul"])

    l_input = InputLayer((None, input_length, nb_words), name="input")
    if mask:
        def construct_mask(X):
            isnt_zero = 1 - (T.eq(X.argmax(axis=2, keepdims=True), 0))
            mask = isnt_zero
            mask = T.set_subtensor(mask[:, 0, 0], 1)
            # mask = 1 if input character isnt the zero character 0 if it is
            # unless it is the first character of the sentence, which is always
            # the zero character, so only the padding zeros (after the end of
            # the sentence) are ignored
            return mask
        l_input_mask = ExpressionLayer(l_input, construct_mask)
    else:
        l_input_mask = None

    l_code = l_input
    for i in range(nb_layers):
        if i == nb_layers - 1:
            only_final = True
        else:
            only_final = False
        l_code = GRULayer(l_code, num_units=hidden,
                          resetgate=Gate(W_cell=None, W_hid=Orthogonal(), nonlinearity=nonlin),
                          updategate=Gate(W_cell=None, W_hid=Orthogonal(), nonlinearity=nonlin),
                          hidden_update=Gate(W_cell=None, W_hid=Orthogonal(), nonlinearity=tanh),
                          mask_input=l_input_mask,
                          grad_clipping=grad_clipping,
                          return_only_final=only_final,
                          name="code")
    l_pre_output = TensorDenseLayer(
        l_code,
        num_units=nb_words,
        name="pre_output",
        nonlinearity=linear,
        b=None,
    )
    l_output = NonlinearityLayer(
        l_pre_output,
        nonlinearity=softmax_,
        name="output"
    )
    layers = [l_input,
              l_code,
              l_pre_output,
              l_output]
    return layers_from_list_to_dict(layers)


def softmax_(x):
    return T.exp(x)/T.exp(x).sum(axis=-1, keepdims=True)
