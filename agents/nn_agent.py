import tensorflow as tf
import numpy as np
import pandas as pd
from random import choice
from chess import Board
from os import listdir
from network import ChessNeuralNetwork


class NeuralNetworkAgent(object):
    def __init__(self, network, name, global_episode_count, checkpoint=None, verbose=False,
                 model_path="/Users/adam/Documents/projects/td_chess/model"):
        self.verbose = verbose
        self.trainer = tf.train.AdamOptimizer()
        self.name = name
        self.checkpoint = checkpoint
        self.model_path = model_path

        self.global_episode_count = global_episode_count

        self.game_turn_count = tf.Variable(0, name='game_turn_count')
        self.game_turn_count_ = tf.placeholder(tf.int32, name='game_turn_count_')
        self.set_game_turn_count_op = tf.assign(self.game_turn_count, self.game_turn_count_, name='set_game_turn_count')
        tf.summary.scalar("game_turn_count", self.game_turn_count)

        self.test_val_ = tf.placeholder(tf.int32, name='test_total_')
        self.test_total = tf.Variable(0, name='test_total')
        self.set_test_total_op = tf.assign(self.test_total, self.test_val_, name='increment_test_total')
        tf.summary.scalar("test_total", self.test_total)

        self.neural_network = network

        self.traces = []
        for var in self.neural_network.trainable_variables:
            trace = np.zeros(var.get_shape())
            self.traces.append(trace)

        self.lamda = .7

        self.grad_vars = self.trainer.compute_gradients(self.neural_network.value, self.neural_network.trainable_variables)

        self.delta_trace_placeholders = [tf.placeholder(tf.float32, shape=var.get_shape(), name=var.op.name+'_PLACEHOLDER') for var in self.neural_network.trainable_variables]
        self.apply_grads = self.trainer.apply_gradients(zip(self.delta_trace_placeholders, self.neural_network.trainable_variables), name='apply_grads')

        with tf.variable_scope('turn_count'):
            self.increment_global_episode_count_op = self.global_episode_count.assign_add(1)

            # self.episode_summary_op = tf.summary.merge(tf.get_collection('episode_summaries'))
            # self.turn_summary_op = tf.summary.merge(tf.get_collection('turn_summaries'))

    def update_traces(self, grads):
        for idx in range(len(grads)):
            self.traces[idx] = self.lamda * self.traces[idx] + grads[idx]

    def reset_traces(self):
        for idx in range(len(self.traces)):
            self.traces[idx][:] = 0

    @staticmethod
    def simple_value(board):
        values = [1, 3, 3, 5, 9]
        s = 0
        for i, v in enumerate(values):
            s += ChessNeuralNetwork.pad_bitmask(board.pieces_mask(i + 1, 1)).sum() * v
            s -= ChessNeuralNetwork.pad_bitmask(board.pieces_mask(i + 1, 0)).sum() * v
        return s

    def train(self, sess, env, num_episode, epsilon, pretrain=False):

        for episode in range(num_episode):
            turn_count = 0
            if self.verbose:
                print(self.name, 'episode:', episode)

            self.reset_traces()
            env.reset()

            while env.get_reward() is None:

                legal_moves = env.get_legal_moves()
                candidate_envs = [env.clone() for _ in legal_moves]
                for idx in range(len(candidate_envs)):
                    candidate_envs[idx].make_move(legal_moves[idx])

                candidate_boards = [candidate_env.board for candidate_env in candidate_envs]

                if pretrain:
                    simple_values = [self.simple_value(board) for board in candidate_boards]
                    candidate_values = [np.tanh(simple_value/5 + .2 * np.random.rand()) for simple_value in simple_values]

                else:
                    for candidate_env, legal_move in zip(candidate_envs, legal_moves):
                        candidate_env.make_move(legal_move)

                    candidate_feature_vectors = np.vstack(
                        [ChessNeuralNetwork.make_feature_vector(candidate_board)
                         for candidate_board in candidate_boards]
                    )
                    candidate_values = sess.run(self.neural_network.value,
                                                feed_dict={
                                                    self.neural_network.feature_vector_: candidate_feature_vectors}
                                                )

                if env.board.turn:
                    move_idx = np.argmax(candidate_values)
                    next_value = np.max(candidate_values)
                else:
                    move_idx = np.argmin(candidate_values)
                    next_value = np.min(candidate_values)

                feature_vector = self.neural_network.make_feature_vector(env.board)
                value, grad_vars = sess.run([self.neural_network.value,
                                             self.grad_vars],
                                            feed_dict={
                                                self.neural_network.feature_vector_: feature_vector,
                                                self.neural_network.target_value_: next_value}
                                            )
                grads, _ = zip(*grad_vars)
                self.update_traces(grads)
                delta = (next_value - value)[0][0]
                sess.run(self.apply_grads,
                         feed_dict={delta_trace_: -delta * trace
                                    for delta_trace_, trace in
                                    zip(self.delta_trace_placeholders, self.traces)}
                         )

                # With probability epsilon ignore the greedy move and make a random move instead. Then reset the traces.
                if np.random.rand() < epsilon:
                    move = choice(env.get_legal_moves())
                    self.reset_traces()
                else:
                    move = legal_moves[move_idx]

                # push the move onto the environment
                env.make_move(move)
                turn_count += 1

            # update traces with final state and reward
            reward = env.get_reward()
            feature_vector = self.neural_network.make_feature_vector(env.board)
            value, grad_vars = sess.run([self.neural_network.value,
                                         self.grad_vars],
                                        feed_dict={
                                            self.neural_network.feature_vector_: feature_vector,
                                            self.neural_network.target_value_: reward}
                                        )
            grads, _ = zip(*grad_vars)
            self.update_traces(grads)
            delta = (reward - value)[0][0]
            sess.run(self.apply_grads,
                     feed_dict={delta_trace_: -delta * trace
                                for delta_trace_, trace in
                                zip(self.delta_trace_placeholders, self.traces)}
                     )
            if self.verbose:
                print("turn count:", turn_count)
                print("global episode count:", sess.run(self.global_episode_count))

            sess.run([self.set_game_turn_count_op,
                      self.increment_global_episode_count_op],
                     feed_dict={self.game_turn_count_: turn_count})

    def test(self, sess, env):
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
                move = self.get_move(sess, env)
                reward = c0.get(board.san(move), 0)
                test_count += 1
                if reward > 0:
                    correct_count += 1
                tot += reward

        sess.run(self.set_test_total_op, feed_dict={self.test_val_: tot})
        global_episode_count = sess.run(self.global_episode_count)
        print("EPISODE", global_episode_count, "TEST TOTAL:", tot)
        return tot


    #  TODO: CALL THIS FUNCTION DURING TRAINING
    def get_move(self, sess, env):
        legal_moves = env.get_legal_moves()

        candidate_envs = [env.clone() for _ in legal_moves]
        for idx in range(len(candidate_envs)):
            candidate_envs[idx].make_move(legal_moves[idx])

        candidate_boards = [candidate_env.board for candidate_env in candidate_envs]

        candidate_feature_vectors = np.vstack(
            [self.neural_network.make_feature_vector(candidate_board)
             for candidate_board in candidate_boards]
        )

        candidate_values = sess.run(self.neural_network.value,
                                    feed_dict={self.neural_network.feature_vector_: candidate_feature_vectors})

        if env.board.turn:
            move_idx = np.argmax(candidate_values)
        else:
            move_idx = np.argmin(candidate_values)

        move = legal_moves[move_idx]

        return move
