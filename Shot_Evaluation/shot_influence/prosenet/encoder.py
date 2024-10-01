"""
Construction function for RNN encoder.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Bidirectional, LSTM, GRU

def rnn(input_shape,
        layer_type='lstm',
        layer_args={},
        layers=[32,64],
        bidirectional=True):
    """
    Recurrent Neural Network encoder constructor function.
    One layer of `layer_type` will be created for each int in `layers`.
    All except the final recurrent layer will return sequences.
    """
    num_layers = len(layers)
    assert num_layers > 0, 'Must have at least one layer'

    layer_fn = GRU if 'gru' in layer_type.lower() else LSTM

    # Construct model
    model = Sequential([InputLayer(input_shape=input_shape)])

    for i, layer_units in enumerate(layers):
        return_seq = False if (i == (num_layers - 1)) else True
        # Temporarilty setting it false
        #return_seq = False

        next_layer = layer_fn(layer_units, return_sequences=return_seq, name=layer_type+str(i), **layer_args)

        if bidirectional:
            model.add(Bidirectional(next_layer))
        else:
            model.add(next_layer)

    return model
