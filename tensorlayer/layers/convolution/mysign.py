import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import binary_weight

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = ['MySign']

class MySign(Layer):
    def __init__(
        self,
        prev_layer,
        name='my_sign'
    ):
        super(MySign, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("MySign")
       
        self.outputs = binary_weight(self.inputs)
        self._add_layers(self.outputs)
    
