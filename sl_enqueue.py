from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from multiprocessing import Process
import time
import tensorflow as tf
import numpy as np
from value_model import ValueModel
import argparse


def get_example(g):
    def f():
        board = g.__next__()
        fv = env.make_feature_vector(board)[0]
        value = env.board_value(board).astype('float32')[0]
        return fv, np.tanh(value/5.0)
    return f


def work(env, job_name, task_index, cluster, log_dir, verbose):

    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)

    if job_name == "ps":
        server.join()
    else:

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:" + job_name + "/task:%d" % task_index,
                cluster=cluster)):

            network = ValueModel(training=True)
            worker_name = 'worker_%03d' % task_index

            summary_op = tf.summary.merge_all()
            is_chief = task_index == 0
            scaffold = tf.train.Scaffold(summary_op=summary_op)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=1,
                                               scaffold=scaffold) as sess:

            agent.sess = sess

            while not sess.should_stop():
                if is_chief:
                    time.sleep(1)
                    episodes_since_apply_grads = sess.run(agent.episodes_since_apply_grad)
                    if episodes_since_apply_grads > 10:
                        episode_number = sess.run(agent.increment_train_episode_count)

                        sess.run(agent.apply_grads, feed_dict={agent.num_grads_: episodes_since_apply_grads})
                        sess.run([agent.reset_episodes_since_apply_grad, agent.reset_grad_accums_op])
                        print(worker_name,
                              "EPISODE:", episode_number,
                              "APPLYING GRADS")
                        print('-' * 100)
                else:
                    episode_number = sess.run(agent.increment_train_episode_count)
                    reward = agent.train(num_moves=10, depth=3, pre_train=False)
                    if agent.verbose:
                        print(worker_name,
                              "EPISODE:", episode_number,
                              "UPDATE:", sess.run(agent.update_count),
                              "REWARD:", reward)
                        print('-' * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("chief_ip")
    parser.add_argument("worker_ip")
    parser.add_argument("tester_ip")

    args = parser.parse_args()

    ps_hosts = [args.chief_ip + ':' + str(2222 + i) for i in range(5)]
    chief_trainer_hosts = [args.chief_ip + ':' + str(3333 + i) for i in range(35)]
    worker_trainer_hosts = [args.worker_ip + ':' + str(3333 + i) for i in range(35)]
    tester_hosts = [args.tester_ip + ':' + str(3333 + i) for i in range(35)]

    ckpt_dir = "./log/" + args.run_name
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ps_hosts,
         "worker": chief_trainer_hosts + worker_trainer_hosts,
         "tester": tester_hosts})
    # cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": chief_trainer_hosts})
    processes = []

    for task_idx, _ in enumerate(ps_hosts):
        env = ChessEnv()
        p = Process(target=work, args=(env, 'ps', task_idx, cluster_spec, ckpt_dir, 1))
        processes.append(p)
        p.start()

    for task_idx, _ in enumerate(chief_trainer_hosts):
        env = ChessEnv()
        p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir, 1))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
