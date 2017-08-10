import tensorflow as tf
from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv, simple_value_from_board
from value_model import ValueModel
import time
import cProfile

def main():

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:

        network = ValueModel(ChessEnv.get_feature_vector_size())
        env = ChessEnv()

        agent = TDLeafAgent('tester_0', network, env, verbose=True)
        agent.sess = sess
        sess.run(tf.global_variables_initializer())

        # cProfile.runctx('agent.test(sess, test_idxs=[0], depth=3)', globals(), locals())
        agent.test(1, depth=1)
        test_results = sess.run(agent.test_results)
        print(sum(test_results))

        # env = Chess(random_position=True, load_pgn=True)

        # for i in range(10):
        #     env.reset()
        #     net_value = sess.run(network.simple_value, feed_dict={network.feature_vector_: env.make_feature_vector2(env.board)})
        #     str_value = simple_value_from_board(env.board)
        #     print(env.board)
        #     print(env.board.fen())
        #     print(net_value, str_value, '\n\n')

if __name__ == "__main__":
    t0 = time.time()
    main()
    print('time:', time.time()-t0)
