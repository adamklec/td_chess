import tensorflow as tf
import multiprocessing
import threading
from time import sleep
from game import Chess
from agents.nn_agent import NeuralNetworkAgent
from network import ChessNeuralNetwork
import multiprocessing

def main():
    load_model = False
    model_path = "/Users/adam/Documents/projects/td_chess/model"

    with tf.Session() as sess:
        global_episode_count = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

        with tf.variable_scope('master'):
            master_netork = ChessNeuralNetwork()

        trainer = tf.train.RMSPropOptimizer(.001)
        num_agents = 2 #multiprocessing.cpu_count()
        agents = []

        for i in range(num_agents):
            agents.append(NeuralNetworkAgent(sess, 'agent_' + str(i), trainer, global_episode_count, verbose=True))

        saver = tf.train.Saver(max_to_keep=5)

        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # agent_threads = []
        pool = multiprocessing.Pool(processes=num_agents)
        # for agent in agents:
        #     agent_train = lambda: agent.train(Chess(), 100, 0.05, saver, pretrain=True)
        #     # t = threading.Thread(target=agent_train)
        #     p = multiprocessing.Process(target=agent_train)
        #     print("starting", agent.name)
        #     p.start()
        #     # p.join()
        #     # sleep(0.5)
        #     agent_threads.append(p)

        agent_train = lambda agent: agent.train(Chess(), 100, 0.05, saver, pretrain=True)
        agent_trains = [agent_train(agent) for agent in agents]
        multiple_results = pool.map(agent_train, agents)

            # coord = tf.train.Coordinator()
        # coord.join(agent_threads)

if __name__ == "__main__":
    main()

