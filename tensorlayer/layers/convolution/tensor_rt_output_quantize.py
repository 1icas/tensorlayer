#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import quantize_active_overflow
from tensorlayer.layers.utils import quantize_weight_overflow

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = ['RTQuanOutput']


class RTQuanOutput(Layer):
    """The :class:`QuanConv2dWithBN` class is a quantized convolutional layer with BN, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.
    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            name='quan_output',
    ):
        super(RTQuanOutput, self
             ).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "RTQuanOutput %s" % (
                self.name
            )
        )
        with tf.variable_scope(name):
            scale = tf.get_variable(
                name = 'scale', shape=(1), dtype=LayersConfig.tf_dtype
            )

            self.outputs = tf.multiply(self.inputs, scale)
            self._add_params(scale)

