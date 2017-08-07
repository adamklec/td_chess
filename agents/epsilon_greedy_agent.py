import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase
import random

class EpsilonGreedyAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):
        super().__init__(name, model, env, verbose)

        tvars = self.model.trainable_variables()
        # grads = tf.gradients([self.model.value], tvars)
        trainer = tf.train.AdamOptimizer()
        grad_vars = trainer.compute_gradients(self.model.value, tvars)
        grads, _ = zip(*grad_vars)
        self.next_value_ = tf.placeholder(tf.float32, name='next_value_')
        delta = self.next_value_ - self.model.value

        traces = []
        update_traces = []
        apply_trace_ops = []
        reset_trace_ops = []

        lamda = tf.constant(0.7, name='lamba')
        lr = tf.constant(0.01, name='lr')

        with tf.variable_scope('traces'):
            for grad, tvar in zip(grads, tvars):

                trace = tf.Variable(tf.zeros(tvar.get_shape()), trainable=False, name='trace')
                traces.append(trace)
                tf.summary.histogram(tvar.name + '_trace', trace)

                update_trace = trace.assign((lamda * trace) + grad)
                update_traces.append(update_trace)

                with tf.control_dependencies([update_trace]):
                    apply_trace_op = tf.assign_add(tvar, lr * delta * trace)
                apply_trace_ops.append(apply_trace_op)

                reset_trace_op = trace.assign(tf.zeros_like(trace))
                reset_trace_ops.append(reset_trace_op)

            self.apply_traces_op = tf.group(*apply_trace_ops)
            self.reset_traces_op = tf.group(*reset_trace_ops)

        with tf.control_dependencies(update_traces):
            self.apply_grads_op = trainer.apply_gradients([(-lr * delta * trace, tvar)
                                                           for trace, tvar in zip(traces, tvars)])

    def train(self, epsilon=.05):
        global_episode_count = self.sess.run(self.global_episode_count)
        self.sess.run(self.increment_global_episode_count_op)

        starting_position_move_str = ','.join([str(m) for m in self.env.get_move_stack()])
        selected_moves = []

        self.env.reset()
        self.sess.run(self.reset_traces_op)

        while self.env.get_reward() is None:
            if random.random() < epsilon:
                move = random.choice(list(self.env.get_legal_moves()))
                self.sess.run(self.reset_traces_op)
            else:
                move, next_value = self.get_move(self.env, return_value=True)
                feature_vector = self.env.make_feature_vector(self.env.board)
                # self.sess.run(self.apply_traces_op, feed_dict={self.model.feature_vector_: feature_vector,
                #                                                self.next_value_: next_value})
                self.sess.run(self.apply_grads_op, feed_dict={self.model.feature_vector_: feature_vector,
                                                              self.next_value_: next_value})
            self.env.make_move(move)

        if self.verbose:
            print("global episode:", global_episode_count,
                  self.name,
                  'reward:', self.env.get_reward())

        selected_moves_string = ','.join([str(m) for m in selected_moves])

        with open("data/move_log.txt", "a") as move_log:
            move_log.write(starting_position_move_str + '/' + selected_moves_string + ':' + str(self.env.get_reward()) + '\n')

    def calculate_candidate_board_values(self, env):
        legal_moves = env.get_legal_moves()
        candidate_boards = []
        for move in legal_moves:
            candidate_board = self.env.board.copy()
            candidate_board.push(move)
            candidate_boards.append(candidate_board)

        feature_vectors = np.vstack([self.env.make_feature_vector(board) for board in candidate_boards])
        values = self.sess.run(self.model.value, feed_dict={self.model.feature_vector_: feature_vectors})
        for idx, board in enumerate(candidate_boards):
            result = board.result()
            if isinstance(result, str):
                result = convert_string_result(result)
            if result is not None:
                values[idx] = result
        return values

    def get_move(self, env, return_value=False):
        values = self.calculate_candidate_board_values(env)
        if env.board.turn:
            value = np.max(values)
            move_idx = np.argmax(values)
        else:
            value = np.min(values)
            move_idx = np.argmin(values)
        move = env.get_legal_moves()[move_idx]
        if return_value:
            return move, value
        else:
            return move


def convert_string_result(string):
    if string == '1-0':
        return 1.0
    elif string == '0-1':
        return -1.0
    elif string == '1/2-1/2':
        return 0.0
    elif string == '*':
        return None
    else:
        raise ValueError('Invalid result encountered')