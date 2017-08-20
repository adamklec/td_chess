import tensorflow as tf


class ValueModel(object):
    def __init__(self, input_dim, hidden_dim=1000):

        with tf.variable_scope('neural_network'):
            self.feature_vector_ = tf.placeholder(tf.float32, shape=[None, input_dim], name='feature_vector_')

            with tf.variable_scope('layer_1'):
                W_1 = tf.get_variable('W_1', initializer=tf.truncated_normal([input_dim, hidden_dim], stddev=0.01))
                self.simple_learned = tf.matmul(self.feature_vector_, W_1)
                hidden = tf.nn.relu(tf.matmul(self.feature_vector_, W_1), name='hidden')
            with tf.variable_scope('layer_2'):
                W_2 = tf.get_variable('W_2', initializer=tf.truncated_normal([hidden_dim, 1], stddev=0.01))
                self.hidden = tf.matmul(hidden, W_2)

            self.value = tf.tanh(self.hidden)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def value_function(self, sess):
        def f(fv):
            value = sess.run(self.value,
                             feed_dict={self.feature_vector_: fv})
            return value
        return f
