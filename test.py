from agents.nn_agent import NeuralNetworkAgent
import tensorflow as tf
from game import Chess


def main():
    with tf.Session() as sess:
        load_model = True
        model_path = "/Users/adam/Documents/projects/td_chess/model"

        trainer = tf.train.AdamOptimizer(1e-4)
        agent = NeuralNetworkAgent(sess, trainer, 'agent_0', test_only=True)
        saver = tf.train.Saver(max_to_keep=5)

        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        env = Chess()
        tot = agent.test(env)
    print(tot)


if __name__ == "__main__":
    main()
