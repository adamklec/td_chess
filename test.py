import tensorflow as tf
from agents.td_leaf_agent import TDLeafAgent
from chess_env import Chess, simple_value_from_board
from value_model import ValueModel
import time
import cProfile

def main():

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:

        network = ValueModel()
        global_episode_count = tf.contrib.framework.get_or_create_global_step()

        agent = TDLeafAgent('tester_0', network, global_episode_count=global_episode_count, load_tests=True)

        sess.run(tf.global_variables_initializer())

        # cProfile.runctx('agent.test(sess, test_idxs=[0], depth=3)', globals(), locals())
        agent.test(sess, depth=4)
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
