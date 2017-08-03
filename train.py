import tensorflow as tf
from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from envs.tic_tac_toe import TicTacToeEnv
from value_model import ValueModel

def main():
    with tf.Session() as sess:
        env = TicTacToeEnv()
        network = ValueModel(env.get_feature_vector_size())
        agent = TDLeafAgent('agent_0', network, env, verbose=True)
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            agent.train(sess, depth=3)

if __name__ == "__main__":
    main()
