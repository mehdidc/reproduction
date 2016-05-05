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

def train_keras_(maxlen, nb_words, cur, prev, next_):
    model = build_model_keras(
        input_length=maxlen,
        nb_words=nb_words, hidden=512)
    loss = {'prev_output': 'categorical_crossentropy',
            'next_output': 'categorical_crossentropy'}
    model.compile(
        optimizer='rmsprop',
        loss=loss
    )
    params = {'input': cur, 'prev_output': prev, 'next_output': next_}

    json_string = model.to_json()
    open('out/arch.json', 'w').write(json_string)
    for i in range(1000):
        model.fit(params, nb_epoch=1)
        model.save_weights('out/weights.h5', overwrite=True)

