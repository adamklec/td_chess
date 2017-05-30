import tensorflow as tf
import chess
import numpy as np


class ChessNeuralNetwork(object):
    def __init__(self, trainer=None):
        with tf.variable_scope('neural_network'):
            self.feature_vector_ = tf.placeholder(tf.float32, shape=[None, 1025], name='feature_vector_')

            with tf.variable_scope('layer_1'):
                W_1 = tf.get_variable('W_1', initializer=tf.truncated_normal([1025, 100], stddev=0.1))
                b_1 = tf.get_variable('b_1', shape=[100], initializer=tf.constant_initializer(0.1))
                relu = tf.nn.relu(tf.matmul(self.feature_vector_, W_1) + b_1, name='relu')

            with tf.variable_scope('layer_2'):
                W_2 = tf.get_variable('W_2', initializer=tf.truncated_normal([100, 1], stddev=0.1))
                b_2 = tf.get_variable('b_2', shape=[1],  initializer=tf.constant_initializer(0.0))
                self.value = tf.nn.tanh(tf.matmul(relu, W_2) + b_2, name='tanh')

            self.target_value_ = tf.placeholder(tf.float32, shape=[], name='target_value_placeholder')

            lamda = tf.constant(0.7, name='lamda')

            traces = []
            update_traces = []
            reset_traces = []

            trace_accums = []
            update_accums = []
            reset_accums = []

            if trainer is not None:
                delta = tf.subtract(self.target_value_, self.value, name='delta')
                trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
                # grads = tf.gradients(self.value, trainable_variables)
                grads_vars = trainer.compute_gradients(self.value, trainable_variables)

                with tf.variable_scope('traces'):
                    # for grad, var in zip(grads, trainable_variables):
                    for grad, var in grads_vars:
                        var_short_name = var.op.name[-3:]
                        with tf.variable_scope(var_short_name):
                            trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                            traces.append(trace)

                            decayed_trace = tf.multiply(lamda, trace, name='decayed_trace')
                            new_trace = tf.add(decayed_trace, grad, name='new_trace')
                            update_trace_op = trace.assign(new_trace)
                            update_traces.append(update_trace_op)

                            reset_trace_op = trace.assign(tf.zeros_like(trace))
                            reset_traces.append(reset_trace_op)

                            trace_accum = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace_accum')
                            trace_accums.append(trace_accum)

                            delta_trace = tf.multiply(tf.reduce_sum(delta), trace, name='delta_trace')
                            update_accum_op = trace_accum.assign_sub(delta_trace)  # Sub for gradient ascent
                            update_accums.append(update_accum_op)

                            reset_accum_op = trace.assign(tf.zeros_like(trace))
                            reset_accums.append(reset_accum_op)

                    self.update_traces_op = tf.group(*update_traces, name='update_traces')
                    with tf.control_dependencies([self.update_traces_op]):
                        self.update_accums_op = tf.group(*update_accums, name='update_accums')

                    self.reset_traces_op = tf.group(*reset_traces, name='reset_traces')
                    self.reset_accums_op = tf.group(*reset_accums, name='reset_accums')

                master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                self.apply_grads = trainer.apply_gradients(zip(trace_accums, master_vars), name='apply_grads')

                for var, trace, grad_accum in zip(tf.trainable_variables(), traces, trace_accums):
                    tf.summary.histogram(var.name, var)
                    tf.summary.histogram(var.name, grad_accum)


    @staticmethod
    def make_feature_vector(board):
        piece_matrix = np.zeros((64, len(chess.PIECE_TYPES) + 1, len(chess.COLORS)))

        # piece positions
        for piece in chess.PIECE_TYPES:
            for color in chess.COLORS:
                piece_matrix[:, piece, int(color)] = ChessNeuralNetwork.pad_bitmask(board.pieces_mask(piece, color))

        # en passant target squares
        if board.ep_square:
            piece_matrix[board.ep_square, len(chess.PIECE_TYPES), int(board.turn)] = 1

        reshaped_piece_matrix = piece_matrix.reshape((64, (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS)))
        feature_array = np.zeros((64, (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS) + 2))
        feature_array[:, :-2] = reshaped_piece_matrix

        # empty squares
        empty_squares = (reshaped_piece_matrix.sum(axis=1) == 0)
        feature_array[empty_squares, :-2] = 1

        # castling rights
        feature_array[:, -1] = ChessNeuralNetwork.pad_bitmask(board.castling_rights)

        feature_vector = np.zeros((1, 1025))
        feature_vector[0, :-1] = np.reshape(feature_array, (1024,))
        feature_vector[0, -1] = board.turn

        return feature_vector

    @staticmethod
    def pad_bitmask(mask):
        mask = [int(s) for s in list(bin(mask)[2:])]
        while len(mask) < 64:
            mask.insert(0, 0)
        return np.array(mask)
