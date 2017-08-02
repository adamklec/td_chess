import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.td_leaf_agent import TDLeafAgent
from envs.tic_tac_toe import TicTacToeEnv
from value_model import ValueModel
import os


def main():
    with tf.Session() as sess:
        env = TicTacToeEnv()
        model = ValueModel(env)
        agent = TDLeafAgent('agent_0', model, env)
        human = HumanAgent()
        sess.run(tf.global_variables_initializer())

        last_log_dir = './log/' + next(os.walk('./log'))[1][-1]
        saver = tf.train.import_meta_graph(last_log_dir + '/model.ckpt-0.meta', clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(last_log_dir))

        players = [human, agent]
        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
