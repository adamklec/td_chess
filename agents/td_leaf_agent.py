import tensorflow as tf
import numpy as np
from anytree import Node
from agents.agent_base import AgentBase


class TDLeafAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        super().__init__(name, model, env, verbose)

        self.ttable = dict()

        with tf.variable_scope('grad_accums'):
            self.grad_accums = [tf.Variable(tf.zeros_like(tvar), trainable=False, name=tvar.op.name)
                                for tvar in self.model.trainable_variables]
            self.grad_accum_s = [tf.placeholder(tf.float32, tvar.get_shape()) for tvar in self.model.trainable_variables]

            self.reset_grad_accums = [tf.assign(grad_accum, tf.zeros_like(tvar))
                                      for grad_accum, tvar in zip(self.grad_accums, self.model.trainable_variables)]
            self.update_grad_accums = [tf.assign_add(grad_accum, grad_accum_)
                                       for grad_accum, grad_accum_ in zip(self.grad_accums, self.grad_accum_s)]

            for grad_accum in self.grad_accums:
                tf.summary.histogram(grad_accum.op.name, grad_accum)

        self.trainer = tf.train.AdamOptimizer()

        self.lamda = .7

        self.grad_vars = self.trainer.compute_gradients(self.model.value, self.model.trainable_variables)
        self.grad_s = [tf.placeholder(tf.float32, shape=var.get_shape(), name=var.op.name+'_PLACEHOLDER')
                       for var in self.model.trainable_variables]
        self.apply_grads = self.trainer.apply_gradients(zip(self.grad_s,
                                                            self.model.trainable_variables),
                                                        name='apply_grads')

    def train(self, depth=1):
        global_episode_count = self.sess.run(self.global_episode_count)
        lamda = 0.7
        self.env.random_position()
        self.ttable = dict()
        starting_position_move_str = ','.join([str(m) for m in self.env.get_move_stack()])
        selected_moves = []

        value_seq = []
        grads_seq = []

        turn_count = 0
        while self.env.get_reward() is None and turn_count < 10:
            move, leaf_value, leaf_node = self.get_move(self.env, depth=depth, return_value_node=True)
            value_seq.append(leaf_value)

            feature_vector = self.env.make_feature_vector(leaf_node.board)
            grad_vars = self.sess.run(self.grad_vars, feed_dict={self.model.feature_vector_: feature_vector})
            grads, _ = zip(*grad_vars)
            grads_seq.append(grads)

            selected_moves.append(move)
            self.env.make_move(move)

            turn_count += 1

        deltas = [value_seq[i+1] - value_seq[i] for i in range(len(value_seq) - 1)]
        grad_accums = [np.zeros_like(grad) for grad in grads_seq[0]]

        for t in range(len(deltas)):
            grads = grads_seq[t]
            inner = sum([lamda ** (j - t) * deltas[j] for j in range(t, len(deltas))])

            for i in range(len(grads)):
                grad_accums[i] -= grads[i] * inner  # subtract for gradient ascent

        self.sess.run(self.update_grad_accums, feed_dict={grad_accum_: grad_accum
                                                          for grad_accum_, grad_accum in zip(self.grad_accum_s,
                                                                                             grad_accums)})
        if global_episode_count % 10 == 0:
            print('EPISODE:', global_episode_count, "updating model")

            self.sess.run(self.apply_grads, feed_dict={grad_: grad_accum
                                                       for grad_, grad_accum in zip(self.grad_s, grad_accums)})
            self.sess.run(self.reset_grad_accums)

        if self.verbose:
            print("global episode:", global_episode_count,
                  self.name,
                  'reward:', self.env.get_reward())

        selected_moves_string = ','.join([str(m) for m in selected_moves])

        with open("data/move_log.txt", "a") as move_log:
            move_log.write(starting_position_move_str + '/' + selected_moves_string + ':' + str(self.env.get_reward()) + '\n')

    def get_move(self, env, depth=3, return_value_node=False):
        self.ttable = dict()
        node = Node('root', board=env.board, move=env.get_null_move())
        leaf_value, leaf_node = self.negamax(node, depth, -1, 1, self.model.value_function(self.sess))

        if len(leaf_node.path) > 1:
            move = leaf_node.path[1].move
        else:
            return env.get_null_move()

        if not node.board.turn:
            leaf_value = -leaf_value

        if return_value_node:
            return move, leaf_value, leaf_node
        else:
            return move

    def get_move_function(self, depth):
        def m(env):
            move = self.get_move(env, depth)
            return move
        return m

    def negamax(self, node, depth, alpha, beta, value_function):
        alpha_orig = alpha

        hash_key = self.env.zobrist_hash(node.board)
        tt_row = self.ttable.get(hash_key)
        if tt_row is not None and tt_row['depth'] >= depth:
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

        elif (depth <= 0 and self.env.is_quiet(node.board)) or depth < -15:
            # print(depth)
            # print(node.path[0].board.fen())
            # print([node.move for node in node.path])
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

        # children = sorted(children, key=lambda child: self.env.move_order_key(child.board, self.ttable))
        children = self.env.sort_children(node, children, self.ttable)
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