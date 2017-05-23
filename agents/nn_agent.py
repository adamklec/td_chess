import tensorflow as tf
import numpy as np
import pandas as pd
from random import choice
from chess import Board
from os import listdir
from network import ChessNeuralNetwork
import time


class NeuralNetworkAgent(object):
    def __init__(self, sess, trainer, name, checkpoint=None, restore=False):
        self.sess = sess
        self.trainer = trainer
        self.name = name
        self.neural_network = ChessNeuralNetwork(self.name)
        self.checkpoint = checkpoint
        if restore:
            self.restore(self.checkpoint)

        with tf.variable_scope(self.name):
            with tf.variable_scope('turn_count'):
                self.game_turn_count = tf.Variable(0, name='game_turn_count', trainable=False, dtype=tf.int32)
                self.global_turn_count = tf.Variable(0, name='global_turn_count', trainable=False, dtype=tf.int32)
                self.increment_turn_count_op = tf.group(self.game_turn_count.assign_add(1), self.global_turn_count.assign_add(1))
                self.reset_game_turn_count_op = self.game_turn_count.assign(0)
                self.reset_global_turn_count_op = self.global_turn_count.assign(0)

            self.target_value_placeholder = tf.placeholder(tf.float32, shape=[],  name='reward_placeholder')

            delta = self.target_value_placeholder - self.neural_network.value

            loss = tf.reduce_sum(tf.square(delta, name='loss'))
            self.average_loss = tf.Variable(0.0, trainable=False)
            loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            average_loss_update_op = tf.group(loss_ema.apply([loss]), self.average_loss.assign(loss_ema.average(loss)))
            tf.summary.scalar('average_loss', self.average_loss)

            grads_and_vars = trainer.compute_gradients(self.neural_network.value, var_list=tf.trainable_variables())

            lamda = tf.constant(0.7, name='lamda')

            traces = []
            update_traces = []
            reset_traces = []

            grad_accums = []
            update_accums = []
            reset_accums = []

            with tf.variable_scope('update_traces'):
                for grad, var in grads_and_vars:
                    if grad is None:
                        grad = tf.zeros_like(var)
                    with tf.variable_scope('trace'):
                        trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                        traces.append(trace)

                        update_trace_op = trace.assign((lamda * trace) + grad)
                        update_traces.append(update_trace_op)

                        reset_trace_op = trace.assign(tf.zeros_like(trace))
                        reset_traces.append(reset_trace_op)

                        grad_accum = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                        grad_accums.append(grad_accum)

                        update_accum_op = grad_accum.assign((-tf.reduce_sum(delta) * trace) + grad_accum)
                        update_accums.append(update_accum_op)

                        reset_accum_op = trace.assign(tf.zeros_like(trace))
                        reset_accums.append(reset_accum_op)

            self.update_traces_op = tf.group(*update_traces)
            with tf.control_dependencies([self.update_traces_op]):
                self.update_accums_op = tf.group(*update_accums)

            self.reset_traces_op = tf.group(*reset_traces)
            self.reset_accums_op = tf.group(*reset_accums)

            master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
            self.apply_grads = trainer.apply_gradients(zip(grad_accums, master_vars))

            for var, trace, grad_accum in zip(tf.trainable_variables(), traces, grad_accums):
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(var.name, grad_accum)
            tf.summary.scalar('average_loss', self.average_loss)

            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
            self.summaries_op = tf.summary.merge_all()

    def train(self, env, num_episode, epsilon):
        tf.train.write_graph(self.sess.graph_def, './model/',
                             'td_chess.pb', as_text=False)

        if self.name == 'agent_0':
            tf.train.write_graph(self.sess.graph_def, './model/', 'td_chess.pb', as_text=False)
            summary_writer = tf.summary.FileWriter('{0}{1}'.format('./log/', int(time.time())), graph=self.sess.graph)

        for episode in range(num_episode):
            # synch network with master
            self.update_target_graph('master', self.name)

            # reset traces and environment
            self.sess.run([self.reset_traces_op, self.reset_game_turn_count_op])
            env.reset()

            while env.get_reward() is None:

                # with probability epsilon apply the grads, reset traces, and make a random move
                if np.random.rand() < epsilon:
                    self.sess.run([self.apply_grads, self.reset_traces_op, self.reset_accums_op, self.increment_turn_count_op])
                    move = choice(env.get_legal_moves())

                # otherwise greedily select a move and update the traces
                else:
                    feature_vector = self.neural_network.make_feature_vector(env.board)
                    legal_moves = env.get_legal_moves()

                    candidate_envs = [env.clone() for _ in legal_moves]
                    for candidate_env, legal_move in zip(candidate_envs, legal_moves):
                        candidate_env.make_move(legal_move)

                    candidate_boards = [candidate_env.board for candidate_env in candidate_envs]

                    candidate_feature_vectors = np.vstack(
                        [self.neural_network.make_feature_vector(candidate_board)
                         for candidate_board in candidate_boards]
                    )

                    candidate_values = self.sess.run(self.neural_network.value,
                                                     feed_dict={
                                                         self.neural_network.feature_vector_placeholder: candidate_feature_vectors})

                    if env.board.turn:
                        move_idx = np.argmax(candidate_values)
                        next_value = np.argmax(candidate_values)
                    else:
                        move_idx = np.argmin(candidate_values)
                        next_value = np.argmin(candidate_values)

                    move = legal_moves[move_idx]
                    self.sess.run([self.update_accums_op, self.increment_turn_count_op],
                                  feed_dict={
                                      self.neural_network.feature_vector_placeholder: feature_vector,
                                      self.target_value_placeholder: next_value})

                # push the move onto the environment
                env.make_move(move)

            # update traces with final state and reward
            feature_vector = self.neural_network.make_feature_vector(env.board)
            self.sess.run([self.update_traces_op, self.apply_grads],
                          feed_dict={
                              self.neural_network.feature_vector_placeholder: feature_vector,
                              self.target_value_placeholder: env.get_reward()})
            print(self.name, "episode:", episode, "turn count:", self.sess.run(self.game_turn_count))
            self.sess.run(self.reset_game_turn_count_op)

            if self.name == 'agent_0':
                summary = self.sess.run(self.summaries_op)
                summary_writer.add_summary(summary, episode)

        if self.name == 'agent_0':
            summary_writer.close()


    def restore(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint('./checkpoints')

        print('Restoring checkpoint: {0}'.format(checkpoint))
        self.saver.restore(self.sess, checkpoint)

    def test(self, env):
        def parse_tests(fn):
            with open(fn, "r") as f:
                tests = f.readlines()

            dicts = []
            data = [[s for s in test.split('; ')] for test in tests]
            for row in data:
                d = dict()
                d['fen'] = row[0].split(' bm ')[0] + " 0 0"
                d['bm'] = row[0].split(' bm ')[1]

                for c in row[1:]:
                    c = c.replace('"', '')
                    c = c.replace(';\n', '')
                    item = c.split(maxsplit=1, sep=" ")
                    d[item[0]] = item[1]
                dicts.append(d)

            for d in dicts:
                move_rewards = {}
                answers = d['c0'].split(',')
                for answer in answers:
                    move_reward = answer.split('=')
                    move_rewards[move_reward[0].strip()] = int(move_reward[1])
                d['c0'] = move_rewards
            df = pd.DataFrame.from_dict(dicts)
            df = df.set_index('id')
            return df

        path = "/Users/adam/Documents/projects/td_chess/STS[1-13]/"
        filenames = [f for f in listdir(path)]
        tot = 0
        test_count = 0
        correct_count = 0
        for filename in filenames:
            df = parse_tests(path + filename)
            for idx, (fen, c0) in enumerate(zip(df.fen, df.c0)):
                board = Board(fen=fen)
                env.reset(board=board)
                move, _ = self.get_move_and_next_value(env)
                reward = c0.get(board.san(move), 0)
                test_count += 1
                if reward > 0:
                    correct_count += 1
                tot += reward
        return tot

    # def get_move_and_next_value(self, env):
    #     legal_moves = env.get_legal_moves()
    #
    #     candidate_envs = [env.clone() for _ in legal_moves]
    #     for candidate_env, legal_move in zip(candidate_envs, legal_moves):
    #         candidate_env.make_move(legal_move)
    #
    #     candidate_boards = [candidate_env.board for candidate_env in candidate_envs]
    #
    #     candidate_feature_vectors = np.vstack(
    #         [self.neural_network.make_feature_vector(candidate_board)
    #          for candidate_board in candidate_boards]
    #     )
    #
    #     candidate_values = self.sess.run(self.neural_network.value,
    #                                      feed_dict={self.neural_network.feature_vector_placeholder: candidate_feature_vectors})
    #
    #     if env.board.turn:
    #         move_idx = np.argmin(candidate_values)
    #         next_value = np.argmin(candidate_values)
    #     else:
    #         move_idx = np.argmax(candidate_values)
    #         next_value = np.argmax(candidate_values)
    #
    #     return legal_moves[move_idx], next_value

    @staticmethod
    def update_target_graph(from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder
