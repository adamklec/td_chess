import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.td_leaf_agent import TDLeafAgent
from envs.tic_tac_toe import TicTacToeEnv
from value_model import ValueModel

def main():
    with tf.Session() as sess:
        env = TicTacToeEnv()
        model = ValueModel()
        agent = TDLeafAgent('agent_0', model, env, sess, restore=True)
        human = HumanAgent()
        players = [human, agent]
        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
