import tensorflow as tf

from agents.nn_agent import NeuralNetworkAgent
from game import Chess

def main():
    with tf.Session() as sess:
        nn_agent = NeuralNetworkAgent(sess)
        env = Chess()
        nn_agent.train(env, 10000, 2, 0.1, verbose=True)

if __name__ == "__main__":
    main()
