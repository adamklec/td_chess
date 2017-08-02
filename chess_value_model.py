import tensorflow as tf
from value_model import ValueModel
from envs.chess import ChessEnv


class ChessValueModel(ValueModel):
    def __init__(self):
        fv_size = ChessEnv.get_feature_vector_size()
        ValueModel.__init__(self, fv_size)

        with tf.variable_scope('neural_network'):
            with tf.variable_scope('simple_value'):
                simple_value_weights = tf.get_variable('simple_value_weights',
                                                       initializer=tf.constant(ChessEnv.get_simple_value_weights(),
                                                                               dtype=tf.float32),
                                                       trainable=False)
        self.simple_hidden = tf.matmul(self.feature_vector_, simple_value_weights)

        self.value = tf.tanh((self.hidden + self.simple_hidden)/5.0)
