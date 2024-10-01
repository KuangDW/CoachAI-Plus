"""
A `Prototypes` Layer and related operations.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer

from prosenet.ops import distance_matrix


class Prototypes(KerasLayer):
    """
    The 'Prototypes Layer' as a tf.keras Layer.
    """
    def __init__(self, k, dmin=1.0, Ld=0.01, Lc=0.01, Le=0.1, **kwargs):
        """
        Parameters
        ----------
        k : int
            Number of prototype vectors to create.
        dmin : float, optional
            Threshold to determine whether two prototypes are close, default=1.0.
            For "diversity" regularization. See paper section 3.2 for details.
        Ld : float, optional
            Weight for "diversity" regularization loss, default=0.01.
        Lc : float, optional
            Weight for "clustering" regularization loss, default=0.01.
        Le : float, optional
            Weight for "evidence" regularization loss, default=0.1.
        **kwargs
            Additional arguments for base `Layer` constructor (name, etc.)
        """
        super(Prototypes, self).__init__(**kwargs)
        self.k = k
        self.dmin = dmin
        self.Ld, self.Lc, self.Le = Ld, Lc, Le


    def build(self, input_shape):
        # Create prototypes as weight variable
        # NOTE: had to add constraint to keep gradients from exploding

        self.d = input_shape[-1]

        # Makes sense to use same `initializer` as LSTM ?
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(1, self.k, self.d),
            initializer='glorot_uniform',
            constraint=lambda w: tf.clip_by_value(w, -1., 1.),
            trainable=True
        )


    def call(self, x, training=None):
        """Forward pass."""

        # L2 distances b/t encodings and prototypes
        x = tf.expand_dims(x, -2)
        d2 = tf.norm(x - self.prototypes, ord=2, axis=-1)
        # Losses only computed `if training`
        if training:
            dLoss = self.Ld * self._diversity_term()
            cLoss = self.Lc * tf.reduce_sum(tf.reduce_min(d2, 0))
            eLoss = self.Le * tf.reduce_sum(tf.reduce_min(d2, 1))
        else:
            dLoss, cLoss, eLoss = 0., 0., 0.

        self.add_loss(dLoss)
        self.add_loss(cLoss, inputs=True)
        self.add_loss(eLoss, inputs=True)

        # Return exponentially squashed distances
        return tf.exp(-d2)


    def _diversity_term(self):
        """Compute the "diversity" loss,
        which penalizes prototypes that are close to each other

        NOTE: Computes full distance matrix, which is redudant, but `prototypes`
              is usually a small-ish tensor and performance is acceptable,
              so I'm not going to worry about it.
        """
        D = distance_matrix(self.prototypes, self.prototypes)

        Rd = tf.nn.relu(-D + self.dmin)

        # Zero the diagonal elements
        zero_diag = tf.ones_like(Rd) - tf.eye(self.k)

        return tf.reduce_sum(tf.square(Rd * zero_diag)) / 2.


    def get_config(self):
        # implement to make serializable
        pass
