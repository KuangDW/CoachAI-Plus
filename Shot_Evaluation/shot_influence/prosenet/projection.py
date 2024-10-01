"""
Implement a PrototypeProjection callback
"""
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class PrototypeProjection(Callback):
    """Implements the "Prototype Projection" step during training.

    The associated `model` must have a sub-model called `encoder`
    and another called `prototypes_layer` (which gets its weights projected)

    Parameters
    ----------
    train_gen : prosenet.DataGenerator
        A generator for data from which to compute encodings --> project
    freq : int, optional
        How often to execute the projection, in epochs, default=4.
    print_argmins : bool, optional
        If True, print indices of closest matches in `train_gen`, default=False.
    """
    def __init__(self, train_gen, freq=4, print_argmins=False, **kwargs):
        self.train_gen = train_gen
        self.freq = freq
        if print_argmins:
            # need to verify behavior of `train_gen`
            raise NotImplementedError()
        super(PrototypeProjection, self).__init__(**kwargs)


    def on_epoch_end(self, epoch, logs=None):
        """
        Prototype projection computation + setting
        """
        if epoch % self.freq == 0:
            print('\nComputing prototype projection...')

            # get encodings of all train sequences
            X_encoded = self.model.encoder.predict_generator(self.train_gen)
            X_encoded = tf.expand_dims(tf.convert_to_tensor(X_encoded), -2)

            # distance matrix from protos
            protos = self.model.prototypes_layer.weights[0]
            d2 = tf.norm(X_encoded - protos, ord=2, axis=-1)

            # reset protos to nearest neighbors
            new_protos = tf.gather(X_encoded, tf.argmin(d2, axis=0), axis=0)
            new_protos = tf.reshape(new_protos, protos.shape) # need to swap axes

            self.model.prototypes_layer.weights[0].assign(new_protos)
            print('... assigned new prototypes from projections.')
