import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.td_leaf_agent import TDLeafAgent
from envs.tic_tac_toe import TicTacToeEnv
from value_model import ValueModel


def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('my_test_model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./log'))
        env = TicTacToeEnv()
        model = ValueModel(env)
        agent = TDLeafAgent('agent_0', model, env)
        human = HumanAgent()
        players = [human, agent]
        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
