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

            # fv_size = env.get_feature_vector_size()
            # network = ValueModel(fv_size)
            network = ChessValueModel()

            opt = tf.train.AdamOptimizer()
            opt = tf.train.SyncReplicasOptimizer(opt, 100)

            worker_name = 'worker_%03d' % task_index
            agent = TDLeafAgent(worker_name,
                                network,
                                env,
                                opt=opt,
                                verbose=verbose)
            summary_op = tf.summary.merge_all()
            is_chief = task_index == 0
            sync_replicas_hook = opt.make_session_run_hook(is_chief)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=1,
                                               hooks=[sync_replicas_hook],
                                               scaffold=tf.train.Scaffold(summary_op=summary_op)) as sess:
            agent.sess = sess

            while not sess.should_stop():
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
    parser.add_argument("idx")
    parser.add_argument("ips", nargs='+')
    args = parser.parse_args()

    ps_hosts = [ip + ':' + str(2222 + i + j*5) for j, ip in enumerate(args.ips) for i in range(5)]
    worker_hosts = [ip + ':' + str(3333 + i + j*40) for j, ip in enumerate(args.ips) for i in range(40)]
    
    ckpt_dir = "./log/" + str(int(time.time()))
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    this_ip = args.ips[int(args.idx)]

    processes = []

    for task_idx, ps_host in enumerate(ps_hosts):
        if this_ip in ps_host:
            p = Process(target=work, args=(None, 'ps', task_idx, cluster_spec, ckpt_dir, 1))
            processes.append(p)
            p.start()
            time.sleep(1)

    for task_idx, worker_host in enumerate(worker_hosts):
        if this_ip in worker_host:
            env = ChessEnv()
            p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir, 1))
            processes.append(p)
            p.start()
            time.sleep(1)

    for process in processes:
        process.join()