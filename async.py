import tensorflow as tf
import multiprocessing
import threading
from time import sleep
from game import Chess
from agents.nn_agent import NeuralNetworkAgent
from network import ChessNeuralNetwork


def main():
    load_model = False
    model_path = "/Users/adam/Documents/projects/td_chess/model"

    with tf.device("/cpu:0"):
        trainer = tf.train.AdamOptimizer(1e-4)
        master_netork = ChessNeuralNetwork('master')

    with tf.Session() as sess:

        agents = []
        num_agents = multiprocessing.cpu_count()

        for i in range(num_agents):
            agents.append(NeuralNetworkAgent(sess, trainer, 'agent_' + str(i)))

        saver = tf.train.Saver(max_to_keep=5)

        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        agent_threads = []
        for agent in agents:
            agent_train = lambda: agent.train(Chess(), 10, 0.05, saver)
            t = threading.Thread(target=agent_train)
            print("starting", agent.name)
            t.start()
            sleep(0.5)
            agent_threads.append(t)

        coord = tf.train.Coordinator()
        coord.join(agent_threads)

if __name__ == "__main__":
    main()

