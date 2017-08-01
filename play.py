import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.td_leaf_agent import TDLeafAgent
from boardgame_envs.chess_env import ChessEnv
import chess


def main():
    with tf.Session() as sess:
        env = ChessEnv()
        # nn_agent = NeuralNetworkAgent(sess, restore=True)
        # players = [nn_agent, nn_agent]
        players = [RandomAgent(), RandomAgent()]
        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
