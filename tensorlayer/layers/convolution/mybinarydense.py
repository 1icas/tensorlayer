import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import binary_weight

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = ['MyBinaryDense']

class MyBinaryDense(Layer):
    def __init__(
        self,
        prev_layer,
        n_units=100,
        act=None,
        W_init=None,
        b_init=None,
        W_init_args=None,
        b_init_args=None,
        name='mybinarydense'):
        super(MyBinaryDense, self).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)  
        logging.info(
            "MyBinaryDense %s: n_unit: %d" % (
                self.name, n_units
            )
        )

        try:
            pre_channel = int(self.inputs.get_shape()[-1])
        except:
            pre_channel = 1
            logging.warning("unknown input channels, set to 1")
        w_shape = (pre_channel, n_units)

        with tf.variable_scope(name):
            W = tf.get_variable(name='binary_dense_w', shape=w_shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args)
            binary_W = binary_weight(W)
            self.outputs = tf.matmul(self.inputs, binary_W, name='mybinarymul')
            if b_init:
                b = tf.get_variable(name='binary_dense_bias', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)
                self.outputs = tf.nn.bias_add(self.outputs, b, name='b_add')
            self.outputs = self._apply_activation(self.outputs)
            self._add_layers(self.outputs)
            if b_init:
                self._add_params([W, b])
            else:
                self._add_params(W)