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

        with tf.variable_scope(self.name):
            with tf.variable_scope('turn_count'):
                self.game_turn_count = tf.Variable(0, name='game_turn_count', trainable=False, dtype=tf.int32)
                self.total_turn_count = tf.Variable(0, name='global_turn_count', trainable=False, dtype=tf.int32)
                self.increment_turn_count_op = tf.group(*[self.game_turn_count.assign_add(1), self.total_turn_count.assign_add(1)], name='increment_turn_count_op')
                self.reset_game_turn_count_op = self.game_turn_count.assign(0)
                self.reset_total_turn_count_op = self.total_turn_count.assign(0)
                self.increment_global_episode_count_op = self.global_episode_count.assign_add(1)

            self.neural_network = ChessNeuralNetwork(trainer=self.trainer, name=self.name)

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
            turn_summary_writer = tf.summary.FileWriter('{0}{1}'.format('./log/turn/', int(time.time())), graph=self.sess.graph)
            episode_summary_writer = tf.summary.FileWriter('{0}{1}'.format('./log/episode/', int(time.time())), graph=self.sess.graph)

        for episode in range(num_episode):
            self.sess.run([self.neural_network.reset_traces_op,
                           self.neural_network.reset_accums_op])
            if self.verbose:
                print(self.name, 'episode:', episode)
            # synch network with master
            # self.update_target_graph('master', self.name)
            self.sess.run(self.neural_network.sync_to_master)

            # reset traces and environment
            self.sess.run([self.neural_network.reset_traces_op, self.reset_game_turn_count_op])
            env.reset()

            while env.get_reward() is None:

                # with probability epsilon apply the grads, reset traces, and make a random move
                if np.random.rand() < epsilon:
                    self.sess.run([self.neural_network.apply_grads,
                                   self.neural_network.reset_traces_op,
                                   self.neural_network.reset_accums_op,
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
                        turn_summaries, _, _ = self.sess.run([self.neural_network.turn_summary_op,
                                                        self.neural_network.update_accums_op,
                                                        self.increment_turn_count_op],
                                                       feed_dict={
                                                           self.neural_network.feature_vector_: feature_vector,
                                                           self.neural_network.target_value_: next_value})
                        # for turn_summary in turn_summaries:
                        #     turn_summary_writer.add_summary(turn_summary, total_turn_count)
                        #     turn_summary_writer.flush()
                        turn_summary_writer.add_summary(turn_summaries, total_turn_count)
                        turn_summary_writer.flush()
                    else:
                        self.sess.run([self.neural_network.update_accums_op,
                                       self.increment_turn_count_op],
                                      feed_dict={
                                          self.neural_network.feature_vector_: feature_vector,
                                          self.neural_network.target_value_: next_value})

                # push the move onto the environment
                env.make_move(move)

            global_episode_count = self.sess.run(self.global_episode_count)
            if self.name == 'agent_0' and global_episode_count % 1 == 0:
                episode_summaries = self.sess.run(self.neural_network.episode_summary_op)
                episode_summary_writer.add_summary(episode_summaries, global_episode_count)
                episode_summary_writer.flush()
                saver.save(self.sess, self.model_path + '/model-' + str(global_episode_count) + '.cptk')

            # update traces with final state and reward
            feature_vector = self.neural_network.make_feature_vector(env.board)
            self.sess.run([self.neural_network.update_traces_op,
                           self.neural_network.apply_grads],
                          feed_dict={
                              self.neural_network.feature_vector_: feature_vector,
                              self.neural_network.target_value_: env.get_reward()})
            if self.verbose:
                print("turn count:", self.sess.run(self.game_turn_count))
                print("global episode count:", self.sess.run(self.global_episode_count))
            self.sess.run([self.reset_game_turn_count_op, self.increment_global_episode_count_op])

        if self.name == 'agent_0':
            turn_summary_writer.close()
            episode_summary_writer.close()

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

    # @staticmethod
    # def update_target_graph(from_scope, to_scope):
    #     from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    #     to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    #     op_holder = []
    #     for from_var, to_var in zip(from_vars, to_vars):
    #         op_holder.append(to_var.assign(from_var))
    #     return op_holder
