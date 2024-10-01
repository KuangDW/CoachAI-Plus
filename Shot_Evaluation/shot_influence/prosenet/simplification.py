"""
Implement 'Prototype Simplfication' as a `Callback`.
"""
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


def best_candidates(S, r, p_i, w=3):
    """
    Parameters
    ----------
    S : iter(seq)

    r : tf.keras.Model
        Sequence encoder model.
    p_i
        Prototype vector to score encodings against.

    Returns
    -------
    _S : iter(seq)
        The `w` sequences with the highest similarity via `r(s) ~ p_i`
    """
    pass


def beam_search(X, r, p_i, w=3):
    """
    Pseudo-code:

    best_cands = lambda S, w : best_candidates(S, r, p_i, w=w)

    S = best_cands(X, w)

    s_opt = None

    while len(S) > 0:
        S_hat = []

        for s in S:
            s_opt = best_cands([s_opt, s], 1)

            if s.size > 0:
                S_hat.extend(sub_sequences(s))

        S = best_cands(S_hat, w)

    return r(s_opt)
    """
    pass


class PrototypeSimplification(Callback):
    """Implements the "Prototype Simplification" algorithm.

    I'm not sure whether or not this is used during training, but I'm going
    to implement it as a `Callback` so that I can test it both ways.

    The associated `model` must have a sub-model called `encoder`
    and another called `prototypes_layer` (which gets its weights projected)

    Parameters
    ----------
    train_gen : prosenet.DataGenerator
        A generator for data in which to search for 'simplified' prototype sequences.
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


    def beam_search(self, w=3):
        """Beam search algorithm. Uses `self.train_gen` as dataset."""
        pass
