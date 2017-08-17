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
        env = ChessEnv(load_tests=True)

        agent = TDLeafAgent('tester_0', network, env, verbose=True)
        agent.sess = sess
        sess.run(tf.global_variables_initializer())

        # cProfile.runctx('agent.test(sess, test_idxs=[0], depth=3)', globals(), locals())
        agent.test(9, depth=1)
        test_results = sess.run(agent.test_results)
        print(sum(test_results))

if __name__ == "__main__":
    t0 = time.time()
    main()
    print('time:', time.time()-t0)
