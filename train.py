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

    # env = ChessEnv(load_tests=True)
    # network = ChessValueModel()
    agent = TDLeafAgent('agent_0', network, env, verbose=True)

    summary_op = tf.summary.merge_all()
    log_dir = "./log/" + str(int(time.time()))

    global_episode_count = tf.train.get_or_create_global_step()
    increment_global_episode_count_op = global_episode_count.assign_add(1)

    with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
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
            sess.run(increment_global_episode_count_op)

if __name__ == "__main__":
    main()
