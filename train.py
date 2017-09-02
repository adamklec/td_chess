import tensorflow as tf
# from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.td_leaf_agent import TDLeafAgent
from envs.tic_tac_toe import TicTacToeEnv
from envs.chess import ChessEnv
from value_model import ValueModel
from chess_value_model import ChessValueModel
import time
import cProfile


def main():
    env = TicTacToeEnv()
    network = ValueModel(env.get_feature_vector_size())
    # agent = EpsilonGreedyAgent('agent_0', network, env, verbose=True)

    # env = ChessEnv()
    # network = ChessValueModel()
    agent = TDLeafAgent('agent_0', network, env, verbose=True)

    summary_op = tf.summary.merge_all()
    log_dir = "./log/" + str(int(time.time()))

    with tf.train.SingularMonitoredSession(checkpoint_dir=log_dir,
                                           scaffold=tf.train.Scaffold(summary_op=summary_op)) as sess:
        agent.sess = sess
        # cProfile.runctx('agent.train(depth=3)', globals(), locals())

        for i in range(10000):
            if i % 100 == 0:
                agent.random_agent_test(depth=2)
                # agent.test(0, depth=3)
                # pass
            else:
                agent.train(depth=2)
            sess.run(agent.increment_episode_count)

if __name__ == "__main__":
    main()
