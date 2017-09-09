import tensorflow as tf
import numpy as np
from anytree import Node
from agents.agent_base import AgentBase
import time

class TDLeafAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 local_model,
                 env,
                 verbose=False):

        super().__init__(name, model, local_model, env, verbose)

        self.episodes_since_apply_grad = tf.Variable(0, trainable=False, name="episodes_since_apply_grad", dtype=tf.int32)
        self.increment_episodes_since_apply_grad = tf.assign_add(self.episodes_since_apply_grad, 1, use_locking=True, name="increment_episodes_since_apply_grad")
        self.reset_episodes_since_apply_grad = tf.assign(self.episodes_since_apply_grad, 0, use_locking=True, name="reset_episodes_since_apply_grad")

        update_grad_accum_ops = []
        reset_grad_accum_ops = []
        self.grad_accums = []
        self.grad_accum_s = []

        for tvar in self.model.trainable_variables:
            grad_accum = tf.Variable(tf.zeros_like(tvar), trainable=False, name=tvar.op.name + "_grad_accum")
            self.grad_accums.append(grad_accum)
            grad_accum_ = tf.placeholder(tf.float32, shape=tvar.get_shape(), name=tvar.op.name + "_grad_accum_")
            self.grad_accum_s.append(grad_accum_)
            update_grad_accum_op = tf.assign_add(grad_accum, grad_accum_)
            update_grad_accum_ops.append(update_grad_accum_op)
            reset_grad_accum_op = tf.assign(grad_accum, tf.zeros_like(tvar))
            reset_grad_accum_ops.append(reset_grad_accum_op)

        self.update_grad_accums_op = tf.group(*update_grad_accum_ops)
        self.reset_grad_accums_op = tf.group(*reset_grad_accum_ops)

        self.opt = tf.train.AdamOptimizer()

        self.grads = tf.gradients(self.local_model.value, self.local_model.trainable_variables)

        self.grad_s = [tf.placeholder(tf.float32, shape=var.get_shape(), name=var.op.name+'_PLACEHOLDER')
                       for var in self.local_model.trainable_variables]
        self.num_grads_ = tf.placeholder(tf.int32, name='num_grads_')
        self.apply_grads = self.opt.apply_gradients(zip([grad_accum/self.num_grads_ for grad_accum in self.grad_accums],
                                                        self.model.trainable_variables),
                                                    name='apply_grads', global_step=self.update_count)

    def train(self, num_moves=10, depth=1):
        t0 = time.time()
        self.sess.run(self.pull_model_op)
        print(self.name, "PULL MODEL TIME:", time.time() - t0)
        lamda = 0.7
        self.env.random_position(episode_count=self.sess.run(self.train_episode_count))
        # self.env.set_board(chess.Board("1k2r3/1p1bP3/2p2p1Q/Ppb5/4Rp1P/2q2N1P/5PB1/6K1 b - - 0 1"))
        self.ttable = dict()
        self.killers = dict()
        # starting_position_move_str = ','.join([str(m) for m in self.env.get_move_stack()])
        # selected_moves = []

        value_seq = []
        grads_seq = []
        traces = [np.zeros(tvar.shape) for tvar in self.local_model.trainable_variables]
        grad_accums = [np.zeros(tvar.shape) for tvar in self.local_model.trainable_variables]

        turn_count = 0

        while self.env.get_reward() is None and turn_count < num_moves:
            move, leaf_value, leaf_node = self.get_move(self.env, depth=depth, return_value_node=True)
            value_seq.append(leaf_value)
            feature_vector = self.env.make_feature_vector2(leaf_node.board)
            grads = self.sess.run(self.grads, feed_dict={self.local_model.feature_vector_: feature_vector})
            grads_seq.append(grads)

            if turn_count > 0:
                delta = (value_seq[-1] - value_seq[-2])[0, 0]
                for grad, trace, grad_accum in zip(grads_seq[-2], traces, grad_accums):
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

        self.sess.run([self.update_grad_accums_op, self.increment_episodes_since_apply_grad],
                      feed_dict={grad_accum_: grad_accum
                                 for grad_accum_, grad_accum in zip(self.grad_accum_s, grad_accums)})

        # selected_moves_string = ','.join([str(m) for m in selected_moves])

        # with open("data/move_log.txt", "a") as move_log:
        #     move_log.write(starting_position_move_str + '/' + selected_moves_string + ':' + str(self.env.get_reward()) + '\n')

        return self.env.get_reward()

    def get_move(self, env, depth=3, return_value_node=False):
        node = Node('root', board=env.board, move=env.get_null_move())
        leaf_value, leaf_node = self.minimax(node, depth, -100000, 100000, self.local_model.value_function(self.sess))
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
            else:
                value = np.array([[value]])
            return value, node

        elif depth <= 0 and self.env.is_quiet(node.board, depth):
            fv = self.env.make_feature_vector2(node.board)
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
        return np.array([[1.0]])
    elif string == '0-1':
        return np.array([[-1.0]])
    elif string == '1/2-1/2':
        return np.array([[0.0]])
    elif string == '*':
        return None
    else:
        raise ValueError('Invalid result encountered')