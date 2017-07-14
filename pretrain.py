import numpy as np
from network import ChessNeuralNetwork
import tensorflow as tf
import pickle
import time

print('loading data...')
[X_train, y_train, X_test, y_test] = pickle.load(open('stockfish_data.pkl', 'rb'))
print('done.')

ckpt_dir = "log/" + str(int(time.time()))

network = ChessNeuralNetwork()
hooks = [tf.train.StopAtStepHook(last_step=10000)]

y_ = tf.placeholder(tf.float32)
l1 = (tf.reduce_sum(tf.abs(network.trainable_variables[0])) + tf.reduce_sum(tf.abs(network.trainable_variables[1])))
mse = tf.reduce_mean((network.combined_score - y_) ** 2)
loss = mse + .001 * l1
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss)
test_writer = tf.summary.FileWriter(ckpt_dir)

for tvar in network.trainable_variables:
    tf.summary.histogram(tvar.op.name, tvar)
tf.summary.scalar('loss', loss)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    print('initializing network...')
    sess.run(tf.global_variables_initializer())
    print('done.')

    for i in range(100000):
        if i % 1000 == 0:
            mse_test = sess.run(loss, feed_dict={network.feature_vector_: X_test, y_: y_test})
            summary, mse_test, pred_vals = sess.run([summary_op, mse, network.learned_value], feed_dict={network.feature_vector_: X_test, y_: y_test})
            test_writer.add_summary(summary, i)
            print(i, 'test rmse:', mse_test**.5)
            # print(pred_vals)

        idxs = np.random.randint(0, len(X_train), 100)
        X_train_batch = X_train[idxs, :]
        y_train_batch = y_train[idxs]
        mse_train, _ = sess.run([mse, train_op], feed_dict={network.feature_vector_: X_train_batch, y_: y_train_batch})
        # print(i, 'train loss:', mse_train)
