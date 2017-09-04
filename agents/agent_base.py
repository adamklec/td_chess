from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import Counter


class AgentBase(metaclass=ABCMeta):

    def __init__(self, name, model, env, verbose=False):

        self.name = name
        self.model = model
        self.env = env
        self.verbose = verbose
        self.sess = None
        self.killers = dict()
        self.ttable = dict()

        if self.model is not None:
            for tvar in self.model.trainable_variables:
                tf.summary.histogram(tvar.op.name, tvar)

            self.update_count = tf.train.get_or_create_global_step()
            self.episode_count = tf.Variable(0, trainable=False)
            self.increment_episode_count = tf.assign_add(self.episode_count, 1, use_locking=True)

            self.test_idx_ = tf.placeholder(tf.int32, name='test_idx_')
            self.test_result_ = tf.placeholder(tf.int32, name='test_result_')

            with tf.name_scope('test_results'):
                self.test_results = tf.Variable(tf.zeros((14,), dtype=tf.int32), name="test_results", trainable=False)
                for i in range(14):
                    tf.summary.scalar("test_" + str(i), tf.reduce_sum(tf.slice(self.test_results, [i], [1])))
                test_total = tf.reduce_sum(self.test_results)
                tf.summary.scalar("test_total", test_total)

                self.update_test_results = tf.scatter_update(self.test_results, self.test_idx_, self.test_result_)

            with tf.name_scope('random_agent_test_results'):

                self.first_player_wins = tf.Variable(0, name="first_player_wins", trainable=False)
                self.first_player_draws = tf.Variable(0, name="first_player_draws", trainable=False)
                self.first_player_losses = tf.Variable(0, name="first_player_losses", trainable=False)

                self.second_player_wins = tf.Variable(0, name="second_player_wins", trainable=False)
                self.second_player_draws = tf.Variable(0, name="second_player_draws", trainable=False)
                self.second_player_losses = tf.Variable(0, name="second_player_losses", trainable=False)

                self.update_first_player_wins = tf.assign(self.first_player_wins, self.test_result_)
                self.update_first_player_draws = tf.assign(self.first_player_draws, self.test_result_)
                self.update_first_player_losses = tf.assign(self.first_player_losses, self.test_result_)

                self.update_second_player_wins = tf.assign(self.second_player_wins, self.test_result_)
                self.update_second_player_draws = tf.assign(self.second_player_draws, self.test_result_)
                self.update_second_player_losses = tf.assign(self.second_player_losses, self.test_result_)

                self.update_random_agent_test_results = [self.update_first_player_wins,
                                                         self.update_first_player_draws,
                                                         self.update_first_player_losses,
                                                         self.update_second_player_wins,
                                                         self.update_second_player_draws,
                                                         self.update_second_player_losses]

                tf.summary.scalar("first_player_wins", self.first_player_wins)
                tf.summary.scalar("first_player_draws", self.first_player_draws)
                tf.summary.scalar("first_player_losses", self.first_player_losses)

                tf.summary.scalar("second_player_wins", self.second_player_wins)
                tf.summary.scalar("second_player_draws", self.second_player_draws)
                tf.summary.scalar("second_player_losses", self.second_player_losses)

    @abstractmethod
    def get_move(self, env):
        return NotImplemented

    @abstractmethod
    def get_move_function(self, depth):
        return NotImplemented

    def test(self, test_idx, depth=1):
        self.killers = dict()
        self.ttable = dict()
        df = self.env.get_test(test_idx)
        total_result = 0
        for i, (_, row) in enumerate(df.iterrows()):
            self.env.make_board(row.fen)
            move = self.get_move(self.env, depth=depth)
            result = row.c0.get(self.env.board.san(move), 0)
            total_result += result
            if self.verbose > 1:
                print(self.name, 'test suite:', test_idx+1, 'row:', i, 'result:', result)
        return total_result

    # def random_agent_test(self, depth=1):
    #
    #     result = self.env.random_agent_test(self.get_move_function(depth))
    #     for update_op, result in zip(self.update_random_agent_test_results, result):
    #         self.sess.run(update_op, feed_dict={self.test_result_: result})
    #
    #     global_episode_count = self.sess.run(self.update_count)
    #
    #     if self.verbose:
    #         print("EPISODE:", global_episode_count, "RANDOM AGENT TEST")
    #         print('FIRST PLAYER:', self.sess.run([self.first_player_wins, self.first_player_draws, self.first_player_losses]))
    #         print('SECOND PLAYER:', self.sess.run([self.second_player_wins, self.second_player_draws, self.second_player_losses]))
    #         print('-' * 100)

    def random_agent_test(self, depth=1):
            x_counter = Counter()
            for _ in range(100):
                self.killers = dict()
                self.ttable = dict()
                reward = self.env.play_random(self.get_move_function(depth), True)
                x_counter.update([reward])

            o_counter = Counter()
            for _ in range(100):
                self.killers = dict()
                self.ttable = dict()
                reward = self.env.play_random(self.get_move_function(depth), False)
                o_counter.update([reward])

            result = [x_counter[1], x_counter[0], x_counter[-1],
                      o_counter[1], o_counter[0], o_counter[-1]]
            return result
