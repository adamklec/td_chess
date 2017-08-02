import tensorflow as tf
import numpy as np
from anytree import Node


class TDLeafAgent(object):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        self.name = name
        self.global_episode_count = tf.contrib.framework.get_or_create_global_step()
        with tf.variable_scope('episode_count'):
            self.increment_global_episode_count_op = self.global_episode_count.assign_add(1)

        self.env = env

        self.verbose = verbose
        self.ttable = dict()

        self.test_score_ = tf.placeholder(tf.float32, name='test_score_')
        self.test_results = tf.Variable(tf.zeros((14,)), name="test_results", trainable=False)
        self.test_idx_ = tf.placeholder(tf.int32, name='test_idx_')
        self.test_result_ = tf.placeholder(tf.float32, name='test_result_')

        self.update_test_results = tf.scatter_update(self.test_results, self.test_idx_, self.test_result_)
        test_total = tf.reduce_sum(self.test_results)
        tf.summary.scalar("test_total", test_total)

        for i in range(14):
            tf.summary.scalar("test_" + str(i), tf.reduce_sum(tf.slice(self.test_results, [i], [1])))

        self.model = model

        with tf.variable_scope('grad_accums'):
            self.grad_accums = [tf.Variable(tf.zeros_like(tvar), trainable=False) for tvar in self.model.trainable_variables]
            self.grad_accum_s = [tf.placeholder(tf.float32, tvar.get_shape()) for tvar in self.model.trainable_variables]

            self.reset_grad_accums = [tf.assign(grad_accum, tf.zeros_like(tvar))
                                      for grad_accum, tvar in zip(self.grad_accums, self.model.trainable_variables)]
            self.update_grad_accums = [tf.assign_add(grad_accum, grad_accum_)
                                      for grad_accum, grad_accum_ in zip(self.grad_accums, self.grad_accum_s)]

        for tvar in self.model.trainable_variables:
            tf.summary.histogram(tvar.op.name, tvar)

        for grad_accum in self.grad_accums:
            tf.summary.histogram(grad_accum.op.name, grad_accum)

        self.trainer = tf.train.AdamOptimizer()

        self.lamda = .7

        self.grad_vars = self.trainer.compute_gradients(self.model.value, self.model.trainable_variables)
        self.grad_s = [tf.placeholder(tf.float32, shape=var.get_shape(), name=var.op.name+'_PLACEHOLDER') for var in self.model.trainable_variables]
        self.apply_grads = self.trainer.apply_gradients(zip(self.grad_s, self.model.trainable_variables), name='apply_grads')

    def train(self, sess, depth=1):
        global_episode_count = sess.run(self.global_episode_count)
        sess.run(self.increment_global_episode_count_op)
        if global_episode_count % 10 == 0:
            run_update = True
        else:
            run_update = False

        lamda = 0.7
        self.env.reset()  # creates random board if env.random_position
        starting_position_move_str = ','.join([str(m) for m in self.env.get_move_stack()])
        selected_moves = []

        value_seq = []
        grads_seq = []
        turn_count = 0

        while self.env.get_reward() is None and turn_count < 10:
            move, leaf_value, leaf_node = self.get_move(sess, self.env, depth)
            selected_moves.append(move)

            feature_vector = self.env.make_feature_vector(leaf_node.board)
            grad_vars = sess.run(self.grad_vars,
                                 feed_dict={self.model.feature_vector_: feature_vector})
            grads, _ = zip(*grad_vars)

            value_seq.append(leaf_value)
            grads_seq.append(grads)

            self.env.make_move(move)
            turn_count += 1

        value_seq.append(value_seq[-1])  # repeating the last value so that delta==0 for the last time step

        deltas = [value_seq[i+1] - value_seq[i] for i in range(len(value_seq) - 1)]
        grad_accums = [np.zeros_like(grad) for grad in grads_seq[0]]

        for t in range(len(grads_seq)):
            grads = grads_seq[t]
            inner = sum([lamda ** (j - t) * deltas[j] for j in range(t, len(grads_seq))])

            for i in range(len(grads)):
                grad_accums[i] -= grads[i] * inner  # subtract for gradient ascent

        sess.run(self.update_grad_accums, feed_dict={grad_accum_: grad_accum
                                                     for grad_accum_, grad_accum in zip(self.grad_accum_s, grad_accums)})

        if run_update:
            print('global_episode_count:', global_episode_count)

            sess.run(self.apply_grads, feed_dict={grad_: grad_accum
                                                  for grad_, grad_accum in zip(self.grad_s, grad_accums)})
            sess.run(self.reset_grad_accums)

        if self.verbose:
            print("global episode:", global_episode_count,
                  self.name,
                  'reward:', self.env.get_reward())

        selected_moves_string = ','.join([str(m) for m in selected_moves])

        with open("data/move_log.txt", "a") as move_log:
            move_log.write(starting_position_move_str + '/' + selected_moves_string + ':' + str(self.env.get_reward()) + '\n')

    def test(self, sess, test_idxs, depth=1):

        from envs.chess import ChessEnv
        from envs.tic_tac_toe import TicTacToeEnv

        if isinstance(self.env, ChessEnv):
            for test_idx in test_idxs:
                result = self.env.test(self.get_move_function(sess, depth), test_idx)
                sess.run(self.update_test_results, feed_dict={self.test_idx_: test_idx,
                                                              self.test_result_: result})
                global_episode_count = sess.run(self.global_episode_count)
                print("EPISODE", global_episode_count,
                      "TEST_IDX", test_idx,
                      "TEST TOTAL:", result)
                print(sess.run(self.test_results))

        elif isinstance(self.env, TicTacToeEnv):
            result = self.env.test(self.get_move_function(sess, depth), None)
            for i, r in enumerate(result):
                sess.run(self.update_test_results, feed_dict={self.test_idx_: i,
                                                              self.test_result_: r})
            global_episode_count = sess.run(self.global_episode_count)
            print("EPISODE", global_episode_count)
            test_results = sess.run(self.test_results)
            print('X:', test_results[:3])
            print('O:', test_results[3:6])

    def load_session(self, sess):
        self.sess_ = sess

    def get_move(self, sess, env, depth):
        self.ttable = dict()
        node = Node('root', board=env.board, move=env.get_null_move())
        leaf_value, leaf_node = self.negamax(node, depth, -1, 1, self.model.value_function(sess))

        if len(leaf_node.path) > 1:
            move = leaf_node.path[1].move
        else:
            return env.get_null_move()

        if node.board.turn:
            return move, leaf_value, leaf_node
        else:
            return move, -leaf_value, leaf_node

    def get_move_function(self, sess, depth):
        def m(env):
            move, value, node = self.get_move(sess, env, depth)
            return move
        return m

    def negamax(self, node, depth, alpha, beta, value_function):
        alpha_orig = alpha

        hash_key = node.board.zobrist_hash()
        tt_row = self.ttable.get(hash_key)
        if tt_row is not None:  # and tt_row['depth'] >= depth:
            if tt_row['flag'] == 'EXACT':
                return tt_row['value'], node
            elif tt_row['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_row['value'])
            elif tt_row['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_row['value'])
            if alpha >= beta:
                return tt_row['value'], node

        if node.board.is_game_over():
            value = node.board.result()
            if isinstance(value, str):
                value = convert_string_result(value)

            if node.board.turn:
                return value, node
            else:
                return -value, node

        # elif depth <= 0 and self.env.is_quiet(node.board):
        elif depth <= 0:
            fv = self.env.make_feature_vector(node.board)
            value = value_function(fv)
            if node.board.turn:
                return value, node
            else:
                return -value, node

        children = []
        for move in node.board.legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            child = Node(str(move), parent=node, board=child_board, move=move)
            children.append(child)

        children = sorted(children, key=lambda child: self.env.move_order_key(child.board, self.ttable))

        v = -100000
        n = node
        for child in children:
            vv, nn = self.negamax(child, depth - 1, -beta, -alpha, value_function)
            vv = -vv
            if vv > v:
                v = vv
                n = nn
            alpha = max(alpha, vv)
            if alpha >= beta:
                break

        if tt_row is None:
            tt_row = dict()
        tt_row['value'] = v
        if v <= alpha_orig:
            tt_row['flag'] = 'UPPERBOUND'
        elif v >= beta:
            tt_row['flag'] = 'LOWERBOUND'
        else:
            tt_row['flag'] = 'EXACT'

        tt_row['depth'] = depth
        self.ttable[hash_key] = tt_row
        return v, n


    def negascout(self, node, depth, alpha, beta, value_function):
        alpha_orig = alpha

        # hash_key = node.board.zobrist_hash()
        # tt_row = self.ttable.get(hash_key)
        # if tt_row is not None and tt_row['depth'] >= depth:
        #     if tt_row['flag'] == 'EXACT':
        #         return tt_row['value'], node
        #     elif tt_row['flag'] == 'LOWERBOUND':
        #         print('found LOWERBOUND')
        #         alpha = max(alpha, tt_row['value'])
        #     elif tt_row['flag'] == 'UPPERBOUND':
        #         print('found UPPERBOUND')
        #         beta = min(beta, tt_row['value'])
        #     if alpha >= beta:
        #         print('alpha:', alpha, 'beta:', beta)
        #         print(tt_row['value'], node)
        #         return tt_row['value'], node

        # if (depth <= 0 and self.env.is_quiet(node.board)) or node.board.is_game_over():
        if depth <= 0 or node.board.is_game_over():
            fv = self.env.make_feature_vector(node.board)
            value = value_function(fv)[0, 0]
            if node.board.turn:
                return value, node
            else:
                return -value, node

        children = []
        for move in node.board.legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            child = Node(str(move), parent=node, board=child_board, move=move)
            children.append(child)

        children = sorted(children, key=lambda child: self.env.move_order_key(child.board, self.ttable))

        n = node
        for idx, child in enumerate(children):
            if idx == 0:
                score, nn = self.negascout(child, depth - 1, -beta, -alpha, value_function)
                score = -score
            else:
                score, nn = self.negascout(child, depth - 1, -alpha - 1, -alpha, value_function)
                score = -score
                if alpha < score < beta:
                    score, nn = self.negascout(child, depth - 1, -beta, -score, value_function)
                    score = -score
            if score > alpha:
                alpha = score
                n = nn
            if alpha >= beta:
                break

        # if tt_row is None:
        #     tt_row = dict()
        # tt_row['value'] = score
        # if score <= alpha_orig:
        #     tt_row['flag'] = 'UPPERBOUND'
        # elif score >= beta:
        #     tt_row['flag'] = 'LOWERBOUND'
        # else:
        #     tt_row['flag'] = 'EXACT'
        #
        # tt_row['depth'] = depth
        # self.ttable[hash_key] = tt_row

        return alpha, n


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