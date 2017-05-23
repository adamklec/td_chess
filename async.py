import tensorflow as tf
import multiprocessing
import threading
from time import sleep
from game import Chess
from agents.nn_agent import NeuralNetworkAgent


def main():
    with tf.device("/cpu:0"):
        trainer = tf.train.AdamOptimizer(1e-4)
        master_agent = NeuralNetworkAgent(None, trainer, 'master')

    with tf.Session() as sess:

        agents = []
        num_agents = multiprocessing.cpu_count()
        for i in range(num_agents):
            agents.append(NeuralNetworkAgent(sess, trainer, 'agent_' + str(i)))

        sess.run(tf.global_variables_initializer())

        agent_threads = []
        for agent in agents:
            agent_train = lambda: agent.train(Chess(), 10, 0.05)
            t = threading.Thread(target=agent_train)
            print("starting", agent.name)
            t.start()
            sleep(0.5)
            agent_threads.append(t)

        coord = tf.train.Coordinator()
        coord.join(agent_threads)

if __name__ == "__main__":
    main()

