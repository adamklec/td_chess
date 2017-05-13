import tensorflow as tf
import numpy as np
import time
import pandas as pd
from random import choice
from chess import Board
from os import listdir


class NeuralNetworkAgent(object):
    def __init__(self, sess, checkpoint=None, restore=False):
        self.sess = sess
        self.checkpoint = checkpoint

        self.batch_size_placeholder = tf.placeholder(tf.float32, shape=[], name='batch_size_placeholder')

        with tf.variable_scope('turn_count'):
            game_turn_count = tf.Variable(0, name='game_turn_count', trainable=False, dtype=tf.int32)
            batch_turn_count = tf.Variable(0, name='batch_turn_count', trainable=False, dtype=tf.int32)
            global_turn_count = tf.Variable(0, name='global_turn_count', trainable=False, dtype=tf.int32)
            self.increment_turn_count_op = tf.group(game_turn_count.assign_add(1),
                                                    batch_turn_count.assign_add(1),
                                                    global_turn_count.assign_add(1))
            self.reset_game_turn_count_op = game_turn_count.assign(0)
            self.reset_batch_turn_count_op = batch_turn_count.assign(0)
            self.reset_global_turn_count_op = global_turn_count.assign(0)

        self.turn_placeholder = tf.placeholder(tf.bool, shape=[], name='turn')
        self.board_placeholder = tf.placeholder(tf.float32, shape=[1, 64, 16], name='board_placeholder')
        self.next_boards_placeholder = tf.placeholder(tf.float32, shape=[None, 64, 16], name='next_boards')

        with tf.variable_scope('feature_vectors'):
            feature_vector = self.make_feature_vectors(self.board_placeholder, self.turn_placeholder)
            feature_vectors = self.make_feature_vectors(self.next_boards_placeholder, tf.logical_not(self.turn_placeholder))

        with tf.variable_scope('state_value_function') as scope:
            value = self.neural_network(feature_vector)
            scope.reuse_variables()
            next_values = self.neural_network(feature_vectors)

        self.next_value = tf.cond(self.turn_placeholder, lambda: tf.reduce_min(next_values), lambda: tf.reduce_max(next_values), name='next_value')
        self.next_board_idx = tf.cond(self.turn_placeholder, lambda: tf.argmin(next_values, axis=0), lambda: tf.argmax(next_values, axis=0), name='next_board_idx')

        self.reward_placeholder = tf.placeholder(tf.float32, shape=[], name='reward_placeholder')

        target_value = tf.cond(tf.shape(self.next_boards_placeholder)[0] > 0, lambda: self.next_value, lambda: self.reward_placeholder, name='target_value')
        delta = tf.sub(target_value, value, name='delta')
        loss = tf.reduce_sum(tf.square(delta, name='loss'))

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        grads_and_vars = opt.compute_gradients(value, var_list=tvars)

        lamda = tf.constant(0.7, name='lamba')
        # tf.summary.scalar('lamda', lamda)

        traces = []
        update_traces = []
        reset_traces = []

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

        self.update_traces_op = tf.group(*update_traces)
        self.reset_traces_op = tf.group(*reset_traces)

        # Average loss
        self.average_loss = tf.Variable(0.0, trainable=False)
        loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        average_loss_update_op = tf.group(loss_ema.apply([loss]),
                                          self.average_loss.assign(loss_ema.average(loss)))
        tf.summary.scalar('average_loss', self.average_loss)

        # Average turn count per game
        average_turn_count_per_game = tf.Variable(0.0, trainable=False)
        turn_count_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        game_turn_count_m1 = tf.cast(game_turn_count, tf.float32) - 1.0
        with tf.control_dependencies([turn_count_ema.apply([game_turn_count_m1])]):
            self.average_turn_count_per_game_update_op = average_turn_count_per_game.assign(turn_count_ema.average(game_turn_count_m1))
        tf.summary.scalar('average_turn_count_per_game', average_turn_count_per_game)

        for var, trace in zip(tvars, traces):
            tf.summary.histogram(var.name, var)

        with tf.control_dependencies([self.increment_turn_count_op,
                                      self.update_traces_op,
                                      average_loss_update_op]):
            grads = []
            for var, trace in zip(tvars, traces):
                grad = -tf.reduce_sum(delta) * trace / tf.cast(batch_turn_count, tf.float32)  # negative sign for gradient ascent
                # tf.summary.histogram(var.name + '/grad', grad)
                grads.append(grad)

            self.train_op = opt.apply_gradients(zip(grads, tvars))

        whitewin = tf.Variable(tf.constant(0.0), name='xwin', trainable=False)
        self.white_win_placeholder = tf.placeholder(tf.float32, shape=[], name='x_win_placeholder')
        self.update_white_win_op = tf.assign(whitewin, self.white_win_placeholder)
        tf.summary.scalar('white_win', whitewin)

        blackwin = tf.Variable(tf.constant(0.0), name='owin', trainable=False)
        self.black_win_placeholder = tf.placeholder(tf.float32, shape=[], name='o_win_placeholder')
        self.update_black_win_op = tf.assign(blackwin, self.black_win_placeholder)
        tf.summary.scalar('black_win', blackwin)

        self.saver = tf.train.Saver(var_list=tvars, max_to_keep=1)

        self.sess.run(tf.global_variables_initializer())

        # add optimizer slot summaries after initialization
        for tvar in tvars:
            tf.summary.histogram(tvar.name, tvar)

            for slot_name in opt.get_slot_names():
                slot = opt.get_slot(tvar, slot_name)
                tf.summary.histogram('{}/{}'.format(tvar.name, slot_name), slot)

        self.summaries_op = tf.summary.merge_all()

        if restore:
            self.restore(self.checkpoint)

    def make_feature_vectors(self, boards, turn):
        with tf.variable_scope('make_feature_vectors'):
            turn = tf.reshape(tf.cast(turn, tf.float32), [-1, 1])
            turns = tf.tile(turn, [1, tf.shape(boards)[0]])
            turns = tf.transpose(turns)

            reshaped_candidate_next_boards = tf.reshape(boards, [tf.shape(boards)[0], 1024])
            feature_vectors = tf.concat(1, [reshaped_candidate_next_boards, turns])

            return feature_vectors

    def neural_network(self, feature_vector):

        with tf.variable_scope("neural_network"):
            with tf.variable_scope('layer_1'):
                W_1 = tf.get_variable('W_1', initializer=tf.truncated_normal([1025, 100], stddev=0.1))
                b_1 = tf.get_variable('b_1', shape=[100], initializer=tf.constant_initializer(-0.1))
                activation_1 = tf.nn.relu(tf.matmul(feature_vector, W_1) + b_1, name='activation_1')

            with tf.variable_scope('layer_2'):
                W_2 = tf.get_variable('W_2', initializer=tf.truncated_normal([100, 1], stddev=0.1))
                b_2 = tf.get_variable('b_2', shape=[1],  initializer=tf.constant_initializer(0.0))
                value = tf.nn.tanh(tf.matmul(activation_1, W_2) + b_2, name='value')

            return value

    def train(self, env, num_epochs, batch_size, epsilon, run_name=None, verbose=False):
        tf.train.write_graph(self.sess.graph_def, './model/', 'td_tictactoe.pb', as_text=False)
        if run_name is None:
            summary_writer = tf.summary.FileWriter('{0}{1}'.format('./log/', int(time.time())), graph=self.sess.graph)
        else:
            summary_writer = tf.summary.FileWriter('{0}{1}'.format('./log/', run_name), graph=self.sess.graph)

        for epoch in range(num_epochs):
            if epoch > 0 and epoch % batch_size == 0:
                if verbose:
                    print('epoch', epoch)
            # reset environment
            self.sess.run([self.reset_traces_op, self.reset_game_turn_count_op])
            env.reset()

            while env.get_reward() is None:
                # get legal moves
                legal_moves = env.get_legal_moves()

                feature_vector = env.get_feature_vector()
                candidate_feature_vectors = [e.get_feature_vector() for e in env.get_candidate_states()]

                # find best move and update trace sum
                move_idx, _ = self.sess.run([self.next_board_idx,
                                             self.train_op],
                                            feed_dict={self.turn_placeholder: env.board.turn,  # not?
                                                       self.board_placeholder: [feature_vector],
                                                       self.next_boards_placeholder: candidate_feature_vectors,
                                                       self.reward_placeholder: 0.0})
                move = legal_moves[move_idx[0]]
                # with probability epsilon
                # make random move and reset traces
                if np.random.rand() < epsilon:
                    move = choice(legal_moves)
                    self.sess.run(self.reset_traces_op)

                # push the move onto the environment
                env.make_move(move)

            # update traces with final state and reward
            feature_vector = env.get_feature_vector()
            self.sess.run([self.train_op,
                           self.average_turn_count_per_game_update_op],
                          feed_dict={self.turn_placeholder: env.board.turn,  # not?
                                     self.board_placeholder: [feature_vector],
                                     self.next_boards_placeholder: np.zeros((0, 64, 16)),
                                     self.reward_placeholder: env.get_reward()})

            if epoch > 0 and epoch % batch_size == 0:
                if verbose:
                    print('loss_ema:', self.sess.run(self.average_loss))

                self.test(env)

                self.saver.save(self.sess, './checkpoints/checkpoint.ckpt')
                summary = self.sess.run(self.summaries_op,
                                        feed_dict={self.batch_size_placeholder: batch_size})
                summary_writer.add_summary(summary, (epoch+1)*batch_size)

            self.sess.run([self.reset_traces_op, self.reset_batch_turn_count_op])

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
                move = self.get_move(env)
                reward = c0.get(board.san(move), 0)
                test_count += 1
                if reward > 0:
                    correct_count += 1
                tot += reward
        return tot

    def get_move(self, env):
        legal_moves = env.get_legal_moves()
        candidate_feature_vectors = [e.get_feature_vector() for e in env.get_candidate_states()]

        move_idx = self.sess.run(self.next_board_idx,
                                 feed_dict={self.turn_placeholder: env.board.turn,
                                            self.next_boards_placeholder: candidate_feature_vectors})
        return legal_moves[move_idx[0]]


