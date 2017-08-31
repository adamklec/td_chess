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

        self.opt = tf.train.AdamOptimizer()

        self.grad_vars = self.opt.compute_gradients(self.model.value, self.model.trainable_variables)
        self.grad_s = [tf.placeholder(tf.float32, shape=var.get_shape(), name=var.op.name+'_PLACEHOLDER')
                       for var in self.model.trainable_variables]
        self.grad_accums = [tf.Variable(np.zeros(tvar.shape), trainable=False, dtype=tf.float32)
                            for tvar in self.model.trainable_variables]
        for tvar, accum in zip(self.model.trainable_variables, self.grad_accums):
            tf.summary.histogram(tvar.op.name + '_accum', accum)

        self.grad_s = [tf.placeholder(tf.float32, shape=tvar.get_shape(), name=tvar.op.name + '_ACCUM_PLACEHOLDER')
                       for tvar in self.model.trainable_variables]
        update_grad_accums = [tf.assign_add(grad_accum, grad_)
                              for grad_accum, grad_ in zip(self.grad_accums, self.grad_s)]
        self.update_grad_accums_op = tf.group(*update_grad_accums)
        self.apply_grads = self.opt.apply_gradients(zip(self.grad_accums,
                                                        self.model.trainable_variables),
                                                    name='apply_grads', global_step=self.episode_count)
        reset_grad_accums = [tf.assign(grad_accum, np.zeros(tvar.shape))
                             for grad_accum, tvar in zip(self.grad_accums, self.model.trainable_variables)]
        self.reset_grad_accums_op = tf.group(*reset_grad_accums)

    def train(self, depth=1):
        global_episode_count = self.sess.run(self.episode_count)
        lamda = 0.7
        self.env.random_position()
        # self.env.set_board(chess.Board("1k2r3/1p1bP3/2p2p1Q/Ppb5/4Rp1P/2q2N1P/5PB1/6K1 b - - 0 1"))
        self.ttable = dict()
        self.killers = dict()
        # starting_position_move_str = ','.join([str(m) for m in self.env.get_move_stack()])
        # selected_moves = []

        value_seq = []
        grads_seq = []
        traces = [np.zeros(tvar.shape) for tvar in self.model.trainable_variables]
        grad_accums = [np.zeros(tvar.shape) for tvar in self.model.trainable_variables]

        turn_count = 0

        while self.env.get_reward() is None and turn_count < 10:
            move, leaf_value, leaf_node = self.get_move(self.env, depth=depth, return_value_node=True)
            value_seq.append(leaf_value)
            feature_vector = self.env.make_feature_vector(leaf_node.board)
            grad_vars = self.sess.run(self.grad_vars, feed_dict={self.model.feature_vector_: feature_vector})
            grads, _ = zip(*grad_vars)
            grads_seq.append(grads)

            if turn_count > 0:
                for grad, trace, grad_accum in zip(grads_seq[-2], traces, grad_accums):
                    delta = value_seq[-1] - value_seq[-2]
                    trace *= lamda
                    trace += grad
                    grad_accum -= delta * trace

                self.env.make_move(move)
            turn_count += 1

            new_killers = dict()
            for killer_depth, killer_list in self.killers.items():
                if killer_depth + 1 < depth:
                    new_killers[killer_depth + 1] = killer_list
            self.killers = new_killers

            self.ttable = {key: row for key, row in self.ttable.items() if row['depth'] + 1 < depth}
            for key, row in self.ttable.items():
                row['depth'] = row['depth'] + 1
                self.ttable[key] = row

        self.sess.run(self.update_grad_accums_op,
                      feed_dict={grad_: grad_accum
                                 for grad_, grad_accum in zip(self.grad_s, grad_accums)})
        if self.verbose:
            print("episode:", global_episode_count,
                  self.name,
                  'reward:', self.env.get_reward())

        if global_episode_count % 100 == 0 and global_episode_count > 0:
            self.sess.run(self.apply_grads)
            self.sess.run(self.reset_grad_accums_op)

        # selected_moves_string = ','.join([str(m) for m in selected_moves])

        # with open("data/move_log.txt", "a") as move_log:
        #     move_log.write(starting_position_move_str + '/' + selected_moves_string + ':' + str(self.env.get_reward()) + '\n')

    def get_move(self, env, depth=3, return_value_node=False):
        node = Node('root', board=env.board, move=env.get_null_move())
        leaf_value, leaf_node = self.minimax(node, depth, -1, 1, self.model.value_function(self.sess))

        if len(leaf_node.path) > 1:
            move = leaf_node.path[1].move
        else:
            return env.get_null_move()

        if return_value_node:
            return move, leaf_value, leaf_node
        else:
            return move

    def get_move_function(self, depth):
        def m(env):
            move = self.get_move(env, depth)
            return move
        return m

    def minimax(self, node, depth, alpha, beta, value_function):

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
                return value, node

        elif depth <= 0 and self.env.is_quiet(node.board, depth):
            fv = self.env.make_feature_vector(node.board)
            value = value_function(fv)
            tt_row = {'value': value, 'flag': 'EXACT', 'depth': depth}
            self.ttable[hash_key] = tt_row
            return value, node

        children = []
        for move in node.board.legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            child = Node(str(move), parent=node, board=child_board, move=move)
            children.append(child)

        children = self.env.sort_children(node, children, self.ttable, self.killers.get(depth, []) + self.killers.get(depth-2, []))

        if node.board.turn:
            best_v = -100000
            best_n = None
            for child in children:
                value, node = self.minimax(child, depth - 1, alpha, beta, value_function)
                if value > best_v:
                    best_v = value
                    best_n = node
                alpha = max(alpha, value)
                if beta <= alpha:
                    if self.killers.get(depth) is None:
                        self.killers[depth] = [child.move, None]
                    else:
                        self.killers[depth] = [child.move, self.killers[depth][0]]
                    break
        else:
            best_v = 100000
            best_n = None
            for child in children:
                value, node = self.minimax(child, depth - 1, alpha, beta, value_function)
                if value < best_v:
                    best_v = value
                    best_n = node
                beta = min(beta, value)
                if beta <= alpha:
                    if self.killers.get(depth) is None:
                        self.killers[depth] = [child.move, None]
                    else:
                        self.killers[depth] = [child.move, self.killers[depth][0]]
                    break

        if tt_row is None:
            tt_row = dict()
        tt_row['value'] = best_v
        if best_v <= alpha_orig:
            tt_row['flag'] = 'UPPERBOUND'
        elif best_v >= beta:
            tt_row['flag'] = 'LOWERBOUND'
        else:
            tt_row['flag'] = 'EXACT'

        tt_row['depth'] = depth
        self.ttable[hash_key] = tt_row

        return best_v, best_n


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