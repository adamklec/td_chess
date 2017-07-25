import tensorflow as tf
from network import ValueNeuralNetwork
from os import listdir
from os.path import isfile, join

path = "/Volumes/Passport/data/"
filenames = [path+f for f in listdir(path) if isfile(join(path, f))][1:]
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.TextLineReader()
key, filename = reader.read(filename_queue)

record_defaults = [[1] for row in range(902)]
data = tf.decode_csv(filename, record_defaults=record_defaults)
X = tf.cast(tf.stack(data[:-1]), tf.float32)
y = tf.cast(data[-1], tf.float32)

# example = ...ops to create one example...
# Create a queue, and an op that enqueues examples one at a time in the queue.
queue = tf.RandomShuffleQueue(1024, 512, [tf.float32, tf.float32], shapes=[(901,), ()])
enqueue_op = queue.enqueue([X, y])
# Create a training graph that starts by dequeuing a batch of examples.
inputs, targets = queue.dequeue_many(512)
network = ValueNeuralNetwork()
loss = tf.reduce_mean((network.value - targets) ** 2)
train_op = tf.train.AdamOptimizer().minimize(loss)


# Create a queue runner that will run 4 threads in parallel to enqueue examples.
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)


# Launch the graph.
with tf.Session() as sess:
    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()

    sess.run(tf.global_variables_initializer())

    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    threads = tf.train.start_queue_runners(coord=coord)
    # Run the training loop, controlling termination with the coordinator.
    for step in range(1000000):
        print(step)
        if coord.should_stop():
            break
        fvs = sess.run(inputs)
        batch_loss, _ = sess.run([loss, train_op], feed_dict={network.feature_vector_: fvs})
        print('batch rmse:', batch_loss**.5)
    # When done, ask the threads to stop.
    coord.request_stop()
    # And wait for them to actually do it.
    coord.join(enqueue_threads)