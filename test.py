import tensorflow as tf
from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from value_model import ValueModel
import time
import cProfile


def main():

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:

        network = ValueModel()
        env = ChessEnv()

        agent = TDLeafAgent('tester_0', network, env, verbose=2)
        agent.sess = sess
        sess.run(tf.global_variables_initializer())

        cProfile.runctx('agent.test(0, depth=3)', globals(), locals())
        # for i in range(14):
        #     result = agent.test(i, depth=2)
        #     print(i, result)
        #     test_results = sess.run(agent.test_results)
        #     print(sum(test_results))

if __name__ == "__main__":
    t0 = time.time()
    main()
    print('time:', time.time()-t0)
