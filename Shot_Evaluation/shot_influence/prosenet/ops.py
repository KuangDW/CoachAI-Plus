"""
Pure TensorFlow functions
"""
import tensorflow as tf


def make2D(t):
    """Make a Tensor `t` 2D, raise ValueError if impossible."""
    ndim = tf.rank(t)
    if ndim == 2:
        return t
    elif ndim == 1:
        return tf.expand_dims(t, 0)
    else:
        t = tf.squeeze(t)
        if tf.rank(t) != 2:
            raise ValueError(f'Tensor cant be made 2D: {t}')
        else:
            return t


def distance_matrix(a, b):
    """Return the distance matrix between rows of `a` and `b`

    They must both be squeezable or expand_dims-able to 2D,
    and have compatible shapes (same number of columns).

    Returns
    -------
    D : Tensor
        2D where D[i, j] == distance(a[i], b[j])
    """
    a_was_b = a is b

    #a = make2D(a)
    rA = tf.expand_dims(tf.reduce_sum(a * a, -1), -1)

    if a_was_b:
        b, rB = a, rA
    else:
        #b = make2D(b)
        rB = tf.expand_dims(tf.reduce_sum(b * b, -1), -1)

    D = rA - 2 * tf.matmul(a, b, transpose_b=True) + tf.transpose(rB)

    return tf.sqrt(D)
