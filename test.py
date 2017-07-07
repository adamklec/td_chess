import tensorflow as tf
from agents.nn_agent import NeuralNetworkAgent
from game import Chess
from network import ChessNeuralNetwork
import time


def main():

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:
        # load_model = False
        # model_path = "/Users/adam/Documents/projects/td_chess/model"

        network = ChessNeuralNetwork()
        global_episode_count = tf.contrib.framework.get_or_create_global_step()

        agent = NeuralNetworkAgent('tester_0', network, global_episode_count=global_episode_count, load_tests=True)
        # saver = tf.train.Saver(max_to_keep=5)
        #
        # if load_model == True:
        #     print('Loading Model...')
        #     ckpt = tf.train.get_checkpoint_state(model_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # else:

        sess.run(tf.global_variables_initializer())

        agent.test(sess, depth=1)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print('time:', time.time()-t0)
