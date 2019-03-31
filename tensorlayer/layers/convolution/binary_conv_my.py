import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import binary_weight

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = ['MyBinaryConv2d']

class MyBinaryConv2d(Layer):
  def __init__(
    self,
    prev_layer,
    n_filter=32,
    filter_size=(3,3),
    strides=(1,1),
    act=None,
    padding='SAME',
    use_gemm=False,
    W_init=None,
    b_init=None,
    W_init_args=None,
    b_init_args=None,
    use_cudnn_on_gpu=None,
    data_format=None,
    name='my_binary_conv2d'
  ) :
    super(MyBinaryConv2d, self).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

    logging.info(
          "MyBinaryConv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
              self.name, n_filter, str(filter_size), str(strides), padding,
              self.act.__name__ if self.act is not None else 'No Activation'
          )
    )

    if use_gemm:
      raise Exception("TODO. The current version use tf.matmul for inferencing.")

    try:
      pre_channel = int(prev_layer.outputs.get_shape()[-1])
    except Exception:
      pre_channel = 1
      logging.warning("unknown input channels, set to 1")

    w_shape = (filter_size[0], filter_size[1], pre_channel, n_filter)
    strides = (1, strides[0], strides[1], 1)

    with tf.variable_scope(name):
      W = tf.get_variable(name='binary_w', shape=w_shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args)
      binary_W = binary_weight(W)
      self.outputs = tf.nn.conv2d(self.inputs, binary_W, strides, padding, use_cudnn_on_gpu,
                data_format)
      if b_init:
        b = tf.get_variable(name='bias', shape=(w_shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)  
        self.outputs = tf.nn.bias_add(self.outputs, b, name='b_add')
      self.outputs = self._apply_activation(self.outputs)
      
      self._add_layers(self.outputs)

      if b_init:
        self._add_params([W, b])
      else:
        self._add_params(W)