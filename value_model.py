import tensorflow as tf
from envs.chess import ChessEnv


class ValueModel:
    def __init__(self, hidden_dim=1000, is_local=False, training=False):
        fv_size = ChessEnv.get_feature_vector_size()
        if is_local:
            collections = [tf.GraphKeys.LOCAL_VARIABLES]
        else:
            collections = None

        with tf.variable_scope('model'):
            if is_local:
                model_type = 'local'
            else:
                model_type = 'global'
            with tf.variable_scope(model_type):

                self.feature_vector_ = tf.placeholder(tf.float32, shape=[None, fv_size], name='feature_vector_')
                self.keep_prob_ = tf.placeholder(tf.float32, name='keep_prob_')

                with tf.variable_scope('layer_1'):
                    W_1 = tf.get_variable('W_1',
                                          shape=[fv_size, hidden_dim],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          collections=collections)
                    hidden_1 = tf.nn.relu(tf.matmul(self.feature_vector_, W_1), name='hidden_1')
                    dropout_1 = tf.layers.dropout(hidden_1, rate=1-self.keep_prob_, training=training, name='dropout_1')

                with tf.variable_scope('layer_2'):
                    W_2 = tf.get_variable('W_2', shape=[hidden_dim, hidden_dim],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          collections=collections)
                    hidden_2 = tf.nn.relu(tf.matmul(dropout_1, W_2), name='hidden_2')
                    dropout_2 = tf.layers.dropout(hidden_2, rate=1-self.keep_prob_, training=training, name='dropout_2')

                with tf.variable_scope('layer_3'):
                    W_3 = tf.get_variable('W_3', shape=[hidden_dim, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          collections=collections)
                    self.value = tf.tanh(tf.matmul(dropout_2, W_3), name='value')

                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope=tf.get_variable_scope().name)

    def value_function(self, sess):
        def f(fv):
            value = sess.run(self.value,
                             feed_dict={self.feature_vector_: fv})
            return value

        return f

