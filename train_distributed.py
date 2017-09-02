from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from envs.tic_tac_toe import TicTacToeEnv
from multiprocessing import Process
import time
import tensorflow as tf
from value_model import ValueModel
from chess_value_model import ChessValueModel


def work(env, job_name, task_index, cluster, log_dir):

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
                                verbose=1)
            summary_op = tf.summary.merge_all()
            is_chief = task_index == 0 and job_name == 'worker'
            sync_replicas_hook = opt.make_session_run_hook(is_chief)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=1,
                                               hooks=[sync_replicas_hook],
                                               scaffold=tf.train.Scaffold(summary_op=summary_op)) as sess:
            agent.sess = sess

            test_period = 30
            if job_name == "worker":
                while not sess.should_stop():
                    episode_count = sess.run(agent.episode_count)
                    sess.run(agent.increment_episode_count)
                    if episode_count % test_period < 2:
                        test_idx = episode_count % test_period
                        print(worker_name, "starting test", test_idx)
                        agent.test(test_idx, depth=2)
                    if episode_count % test_period < 14:
                        # agent.random_agent_test(depth=3)

                    else:
                        agent.train(depth=2)

if __name__ == "__main__":
    ps_hosts = ['localhost:' + str(2222 + i) for i in range(5)]
    worker_hosts = ['localhost:' + str(3333 + i) for i in range(30)]
    ckpt_dir = "./log/" + str(int(time.time()))
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    processes = []

    for task_idx, _ in enumerate(ps_hosts):
        p = Process(target=work, args=(None, 'ps', task_idx, cluster_spec, ckpt_dir,))
        processes.append(p)
        p.start()
        time.sleep(1)

    for task_idx, _ in enumerate(worker_hosts):
        env = ChessEnv()
        # env = TicTacToeEnv()
        p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir,))
        processes.append(p)
        p.start()
        time.sleep(1)

    for process in processes:
        process.join()
