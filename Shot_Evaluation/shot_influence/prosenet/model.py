"""
A `ProSeNet` Model and related operations.
"""
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from prosenet import encoder, Prototypes


#########################
### Default arguments ###
#########################

# Change these later probably
default_rnn_args = {
    'layer_type' : 'lstm',
    'layer_args' : {},
    'layers' : [32,64],
    'bidirectional' : True
}

default_prototypes_args = {
    'dmin' : 1.0,
    'Ld' : 0.01,
    'Lc' : 0.01,
    'Le' : 0.1
}


class ProSeNet(tf.keras.Model):

    def __init__(
        self,
        input_shape,
        nclasses,
        k,
        rnn_args=default_rnn_args,
        prototypes_args=default_prototypes_args,
        L1=0.1
    ):
        """
        Parameters
        ----------
        input_shape : tuple(int)
            Shape of input sequences, probably: (length, features)
            `length` may be `None` for variable length data.
        nclasses : int
            Number of output classes
        k : int
            Number of prototype vectors in `Prototypes` layer
        rnn_args : dict, optional
            Any updates to default `encoder.rnn` construction args.
        prototypes_args : dict, optional
            Any updates to default `Prototypes` layer args.
        L1 : float, optional
            Strength of L1 regularization for `Dense` classifier kernel.
        """
        super(ProSeNet, self).__init__()

        # Construct encoder network
        default_rnn_args.update(rnn_args)
        self.encoder = encoder.rnn(input_shape, **default_rnn_args)

        # Construct `Prototypes` layer
        default_prototypes_args.update(prototypes_args)
        self.prototypes_layer = Prototypes(k, **default_prototypes_args)

        # Dense classifier with kernel restricted to >= 0.
        self.classifier = layers.Dense(
            nclasses,
            activation='softmax',
            use_bias=False,
            kernel_regularizer=regularizers.l1(l=L1),
            kernel_constraint=lambda w: tf.nn.relu(w),
            name='classifier'
        )


    def call(self, x):
        """Full forward call."""
        a = self.similarity_vector(x)

        return self.classifier(a)


    def similarity_vector(self, x):
        """Return the similarity vector(s) of shape (batches, k,)."""

        r_x = self.encoder(x)

        return self.prototypes_layer(r_x)
