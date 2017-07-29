import tensorflow as tf

from agents.nn_agent import NeuralNetworkAgent
from boardgame_envs.chess_env import ChessEnv
from boardgame_envs.tic_tac_toe_env import TicTacToeEnv
from network import ValueNeuralNetwork

def main():
    with tf.Session() as sess:
        global_episode_count = tf.contrib.framework.get_or_create_global_step()
        network = ValueNeuralNetwork()
        env = TicTacToeEnv(random_position=True)
        agent = NeuralNetworkAgent('agent_0', network, env, global_episode_count, verbose=True)
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            agent.train(sess, depth=3)

if __name__ == "__main__":
    main()
