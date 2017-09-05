import tensorflow as tf
from envs.chess import ChessEnv


class ChessValueModel:
    def __init__(self):
        fv_size = ChessEnv.get_feature_vector_size()
        simple_value_weights = ChessEnv.get_material_value_weights()
        with tf.variable_scope('neural_network'):
            with tf.variable_scope('simple_value_weights'):
                simple_value_weights = tf.get_variable('simple_value_weights',
                                                       initializer=tf.constant(simple_value_weights, dtype=tf.float32),
                                                       trainable=False)

            self.feature_vector_ = tf.placeholder(tf.float32, shape=[None, fv_size], name='feature_vector_')
            with tf.variable_scope('layer_1'):
                W_1 = tf.get_variable('W_1', shape=[fv_size, 1000], initializer=tf.contrib.layers.xavier_initializer())
                hidden_1 = tf.nn.relu(tf.matmul(self.feature_vector_, W_1), name='hidden')

            with tf.variable_scope('layer_2'):
                W_2 = tf.get_variable('W_2', shape=[1000, 1000], initializer=tf.contrib.layers.xavier_initializer())
                hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W_2), name='hidden')

            with tf.variable_scope('layer_3'):
                W_3 = tf.get_variable('W_3', shape=[1000, 1], initializer=tf.contrib.layers.xavier_initializer())
                learned_value = tf.matmul(hidden_2, W_3)

            self.simple_value = tf.matmul(self.feature_vector_, simple_value_weights) / 5.0

            self.value = tf.tanh(self.simple_value + learned_value)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                         scope=tf.get_variable_scope().name)

    def value_function(self, sess):
        def f(fv):
            value = sess.run(self.value,
                             feed_dict={self.feature_vector_: fv})
            return value

        return f

