import tensorflow as tf
import numpy as np
import pandas as pd
from random import choice
from chess import Board
from os import listdir
from network import ChessNeuralNetwork
import time


class NeuralNetworkAgent(object):
    def __init__(self, sess, name, global_episode_count, checkpoint=None, verbose=False,
                 model_path="/Users/adam/Documents/projects/td_chess/model"):
        self.verbose = verbose
        self.sess = sess
        self.trainer = tf.train.AdamOptimizer()
        self.name = name
        self.checkpoint = checkpoint
        self.model_path = model_path
        self.global_episode_count = global_episode_count

        self.master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')

        with tf.variable_scope(self.name):
            self.neural_network = ChessNeuralNetwork(name=self.name)

            traces = []
            update_traces = []
            reset_traces = []

            trace_accums = []
            update_accums = []
            reset_accums = []

            lamda = tf.constant(0.7, name='lamda')

            with tf.variable_scope('traces'):
                delta = tf.subtract(self.neural_network.target_value_, self.neural_network.value, name='delta')

                grads_vars = self.trainer.compute_gradients(self.neural_network.value,
                                                            self.neural_network.trainable_variables)
                for grad, var in grads_vars:
                    var_short_name = var.op.name[-3:]
                    with tf.variable_scope(var_short_name):
                        trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                        traces.append(trace)

                        update_trace_op = trace.assign(lamda * trace + grad)
                        update_traces.append(update_trace_op)

                        reset_trace_op = trace.assign(tf.zeros_like(trace))
                        reset_traces.append(reset_trace_op)

                        trace_accum = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace_accum')
                        trace_accums.append(trace_accum)

                        delta_trace = tf.multiply(tf.reduce_sum(delta), trace, name='delta_trace')
                        update_accum_op = trace_accum.assign_sub(delta_trace)  # sub for gradient ascent
                        update_accums.append(update_accum_op)

                        reset_accum_op = trace.assign(tf.zeros_like(trace))
                        reset_accums.append(reset_accum_op)

                self.update_traces_op = tf.group(*update_traces, name='update_traces')
                with tf.control_dependencies([self.update_traces_op]):
                    self.update_accums_op = tf.group(*update_accums, name='update_accums')

                self.reset_traces_op = tf.group(*reset_traces, name='reset_traces')
                self.reset_accums_op = tf.group(*reset_accums, name='reset_accums')

            self.apply_grads = self.trainer.apply_gradients(zip(trace_accums, self.master_vars), name='apply_grads')

            with tf.variable_scope('sync_to_master'):
                sync_ops = []
                for master_var, trainable_var in zip(self.master_vars, self.neural_network.trainable_variables):
                    sync_ops.append(trainable_var.assign(master_var))
                self.sync_to_master = tf.group(*sync_ops, name='sync_to_master')

            with tf.variable_scope('turn_count'):
                self.game_turn_count = tf.Variable(0, name='game_turn_count', trainable=False, dtype=tf.int32)
                self.total_turn_count = tf.Variable(0, name='global_turn_count', trainable=False, dtype=tf.int32)
                self.increment_turn_count_op = tf.group(*[self.game_turn_count.assign_add(1), self.total_turn_count.assign_add(1)], name='increment_turn_count_op')
                self.reset_game_turn_count_op = self.game_turn_count.assign(0)
                self.reset_total_turn_count_op = self.total_turn_count.assign(0)
                self.increment_global_episode_count_op = self.global_episode_count.assign_add(1)

            if name == 'agent_0':
                for master_var in self.master_vars:
                    tf.summary.histogram(master_var.op.name, master_var, collections=['episode_summaries'])
                for var, trace, grad_accum in zip(self.neural_network.trainable_variables, traces, trace_accums):
                    tf.summary.histogram(var.op.name, var, collections=['episode_summaries'])
                    tf.summary.histogram(trace.op.name, trace, collections=['episode_summaries'])
                    tf.summary.histogram(grad_accum.op.name, grad_accum, collections=['episode_summaries'])
                for grad, _ in grads_vars:
                    tf.summary.histogram(grad.op.name, grad, collections=['turn_summaries'])

            self.episode_summary_op = tf.summary.merge(tf.get_collection('episode_summaries'))
            self.turn_summary_op = tf.summary.merge(tf.get_collection('turn_summaries'))

    @staticmethod
    def simple_value(board):
        values = [1, 3, 3, 5, 9]
        s = 0
        for i, v in enumerate(values):
            s += ChessNeuralNetwork.pad_bitmask(board.pieces_mask(i + 1, 1)).sum() * v
            s -= ChessNeuralNetwork.pad_bitmask(board.pieces_mask(i + 1, 0)).sum() * v
        return s

    def train(self, env, num_episode, epsilon, saver, pretrain=False):

        if self.name == 'agent_0':
            tf.train.write_graph(self.sess.graph_def, './model/', 'td_chess.pb', as_text=False)
            summary_writer = tf.summary.FileWriter('{0}{1}'.format('./log/', int(time.time())), graph=self.sess.graph)

        for episode in range(num_episode):
            self.sess.run([self.reset_traces_op,
                           self.reset_accums_op])
            if self.verbose:
                print(self.name, 'episode:', episode)

            # sync network with master
            self.sess.run(self.sync_to_master)

            # reset traces and environment
            self.sess.run([self.reset_traces_op, self.reset_game_turn_count_op])
            env.reset()

            while env.get_reward() is None:

                # with probability epsilon apply the grads, reset traces, and make a random move
                if np.random.rand() < epsilon:
                    self.sess.run([self.apply_grads,
                                   self.reset_traces_op,
                                   self.reset_accums_op,
                                   self.increment_turn_count_op])
                    move = choice(env.get_legal_moves())

                # otherwise greedily select a move and update the traces
                else:
                    feature_vector = self.neural_network.make_feature_vector(env.board)
                    legal_moves = env.get_legal_moves()

                    candidate_envs = [env.clone() for _ in legal_moves]
                    for candidate_env, legal_move in zip(candidate_envs, legal_moves):
                        candidate_env.make_move(legal_move)

                    candidate_boards = [candidate_env.board for candidate_env in candidate_envs]

                    if pretrain:
                        simple_values = [self.simple_value(board) for board in candidate_boards]
                        candidate_values = [np.tanh(simple_value/5 + np.random.rand()) for simple_value in simple_values]
                    else:
                        candidate_feature_vectors = np.vstack(
                            [ChessNeuralNetwork.make_feature_vector(candidate_board)
                             for candidate_board in candidate_boards]
                        )
                        candidate_values = self.sess.run(self.neural_network.value,
                                                         feed_dict={
                                                             self.neural_network.feature_vector_: candidate_feature_vectors})

                    if env.board.turn:
                        move_idx = np.argmax(candidate_values)
                        next_value = np.max(candidate_values)
                    else:
                        move_idx = np.argmin(candidate_values)
                        next_value = np.min(candidate_values)

                    move = legal_moves[move_idx]
                    total_turn_count = self.sess.run(self.total_turn_count)

                    if self.name == 'agent_0' and total_turn_count % 100 == 0:
                        turn_summaries, _, _ = self.sess.run([self.turn_summary_op,
                                                              self.update_accums_op,
                                                              self.increment_turn_count_op],
                                                             feed_dict={
                                                                 self.neural_network.feature_vector_: feature_vector,
                                                                 self.neural_network.target_value_: next_value})
                        summary_writer.add_summary(turn_summaries, total_turn_count)
                        summary_writer.flush()
                    else:
                        self.sess.run([self.update_accums_op,
                                       self.increment_turn_count_op],
                                      feed_dict={
                                          self.neural_network.feature_vector_: feature_vector,
                                          self.neural_network.target_value_: next_value})

                # push the move onto the environment
                env.make_move(move)

            global_episode_count = self.sess.run(self.global_episode_count)

            if self.name == 'agent_0' and global_episode_count % 1 == 0:
                episode_summaries = self.sess.run(self.episode_summary_op)
                summary_writer.add_summary(episode_summaries, global_episode_count)
                summary_writer.flush()
                saver.save(self.sess, self.model_path + '/model-' + str(global_episode_count) + '.cptk')

            # update traces with final state and reward
            feature_vector = self.neural_network.make_feature_vector(env.board)
            self.sess.run([self.update_traces_op,
                           self.apply_grads],
                          feed_dict={
                              self.neural_network.feature_vector_: feature_vector,
                              self.neural_network.target_value_: env.get_reward()})
            if self.verbose:
                print("turn count:", self.sess.run(self.game_turn_count))
                print("global episode count:", self.sess.run(self.global_episode_count))
            self.sess.run([self.reset_game_turn_count_op, self.increment_global_episode_count_op])

        if self.name == 'agent_0':
            summary_writer.close()
            summary_writer.close()

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
                move = self.get_move(env)
                reward = c0.get(board.san(move), 0)
                test_count += 1
                if reward > 0:
                    correct_count += 1
                tot += reward
        return tot

    def get_move(self, env):
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
                                         feed_dict={self.neural_network.feature_vector_: candidate_feature_vectors})

        if env.board.turn:
            move_idx = np.argmin(candidate_values)
        else:
            move_idx = np.argmax(candidate_values)

        move = legal_moves[move_idx]

        return move
