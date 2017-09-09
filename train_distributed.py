from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from multiprocessing import Process
import time
import tensorflow as tf
from chess_value_model import ChessValueModel
import argparse


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

            with tf.device("/job:worker/task:%d/cpu:0" % task_index):
                with tf.variable_scope('local'):
                    local_network = ChessValueModel(is_local=True)

            # fv_size = env.get_feature_vector_size()
            # network = ValueModel(fv_size)
            network = ChessValueModel()

            worker_name = 'worker_%03d' % task_index
            agent = TDLeafAgent(worker_name,
                                network,
                                local_network,
                                env,
                                verbose=verbose)
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
                    time.sleep(5)
                    episodes_since_apply_grads = sess.run(agent.episodes_since_apply_grad)
                    if episodes_since_apply_grads > 10:
                        episode_number = sess.run(agent.increment_train_episode_count)

                        t0 = time.time()
                        sess.run(agent.apply_grads, feed_dict={agent.num_grads_: episodes_since_apply_grads})
                        sess.run([agent.reset_episodes_since_apply_grad, agent.reset_grad_accums_op])
                        print(worker_name,
                              "EPISODE:", episode_number,
                              "APPLYING GRADS:", time.time() - t0)
                        print('-' * 100)
                else:
                    episode_number = sess.run(agent.increment_train_episode_count)
                    reward = agent.train(num_moves=10, depth=3)
                    if agent.verbose:
                        print(worker_name,
                              "EPISODE:", episode_number,
                              "UPDATE:", sess.run(agent.update_count),
                              "REWARD:", reward)
                        print('-' * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("ips", nargs='+')
    args = parser.parse_args()
    this_ip = args.ips[0]
    that_ip = args.ips[1]

    ps_hosts = [this_ip + ':' + str(2222 + i) for i in range(5)] #+ [that_ip + ':' + str(2222 + i) for i in range(5)]
    # ps_hosts = [this_ip + ':2222']
    worker_hosts = [this_ip + ':' + str(3333 + i) for i in range(40)] + [that_ip + ':' + str(3333 + i) for i in range(40)]

    ckpt_dir = "./log/" + args.run_name
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    processes = []

    for task_idx, ps_host in enumerate(ps_hosts):
        if this_ip in ps_host:
            p = Process(target=work, args=(None, 'ps', task_idx, cluster_spec, ckpt_dir, 1))
            processes.append(p)
            p.start()

    for task_idx, worker_host in enumerate(worker_hosts):
        if this_ip in worker_host:
            env = ChessEnv()
            p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir, 1))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()