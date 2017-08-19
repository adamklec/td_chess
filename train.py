import tensorflow as tf
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from envs.tic_tac_toe import TicTacToeEnv
from value_model import ValueModel
import time


def main():
    env = TicTacToeEnv()
    network = ValueModel(env.get_feature_vector_size())
    agent = EpsilonGreedyAgent('agent_0', network, env, verbose=True)
    summary_op = tf.summary.merge_all()
    log_dir = "./log/" + str(int(time.time()))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                           scaffold=tf.train.Scaffold(summary_op=summary_op)) as sess:
        agent.sess = sess

        for i in range(10000000):
            if i % 1000 == 0:
                agent.random_agent_test()
            agent.train()

if __name__ == "__main__":
    main()
