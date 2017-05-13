from agents.nn_agent import NeuralNetworkAgent
import tensorflow as tf
from game import Chess


def main():
    with tf.Session() as sess:
        agent = NeuralNetworkAgent(sess, restore=True)
        env = Chess()
        tot = agent.test(env)
    print(tot)


if __name__ == "__main__":
    main()
