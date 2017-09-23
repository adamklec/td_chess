import numpy as np
import tensorflow as tf
import time

from envs.chess import ChessEnv
from value_model import ValueModel


def get_example(g):
    def f():
        board = g.__next__()
        fv = env.make_feature_vector(board)[0]
        value = env.board_value(board).astype('float32')[0]
        return fv, np.tanh(value/5.0)
    return f


env = ChessEnv()

fv_size = env.get_feature_vector_size()
queue = tf.FIFOQueue(1000, [tf.float32, tf.float32], shapes=[[fv_size], [1]])
num_threads = 4

enqueue_ops = [queue.enqueue(tf.py_func(get_example(ChessEnv.random_board_generator2()), [], (tf.float32, tf.float32)))
               for _ in range(num_threads)]

queue_count_op = queue.size()
queue_count = tf.Variable(0, trainable=False, name='queue_count')
update_queue_count = queue_count.assign(queue_count_op)
tf.summary.scalar('queue_count', queue_count)

qr = tf.train.QueueRunner(queue=queue, enqueue_ops=enqueue_ops)
tf.train.add_queue_runner(qr)

batch_size = 100
get_batch = queue.dequeue_many(batch_size)

model = ValueModel(training=True)

for tvar in model.trainable_variables:
    tf.summary.histogram(tvar.op.name, tvar)

target_ = tf.placeholder(tf.float32, shape=[None, 1])

loss = tf.reduce_mean(tf.square(target_ - model.value, name='loss'))

rmse_train = tf.Variable(0.0, trainable=False, name='mse_train')
update_mse_train = rmse_train.assign(loss ** .5)
tf.summary.scalar('rmse_train', rmse_train)

rmse_predict = tf.Variable(0.0, trainable=False, name='rmse_predict')
update_rmse_predict = rmse_predict.assign(loss ** .5)
tf.summary.scalar('rmse_predict', rmse_predict)

opt = tf.train.AdamOptimizer()
global_step = tf.train.get_or_create_global_step()
train_op = opt.minimize(loss, global_step=global_step)

summary_op = tf.summary.merge_all()

scaffold = tf.train.Scaffold(summary_op=summary_op)
log_dir = "./log/" + str(int(time.time()))

with tf.train.MonitoredTrainingSession(scaffold=scaffold,
                                       checkpoint_dir=log_dir,
                                       save_summaries_steps=10,
                                       log_step_count_steps=10
                                       ) as sess:
    while not sess.should_stop():
        X, y = sess.run(get_batch)
        _, loss_predict, = sess.run([loss, update_rmse_predict], feed_dict={model.feature_vector_: X, target_: y, model.keep_prob_: 1.0})
        y_pred, _, loss_train = sess.run([model.value, train_op, update_mse_train], feed_dict={model.feature_vector_: X, target_: y, model.keep_prob_: 0.5})

        batch_idx = sess.run(global_step)
        if batch_idx % 10 == 0:
            q_count = sess.run(update_queue_count)
            print(batch_idx, loss_train, loss_predict, q_count)
            # for row in zip(5.0 * np.arctanh(y)[:, 0], y[:, 0], y_pred[:, 0], y[:, 0] - y_pred[:, 0]):
            #     print(row)
            #
            # print('rmse:', np.mean(np.square(y[:, 0] - y_pred[:, 0])) ** .5)
            # print('-' * 100)
