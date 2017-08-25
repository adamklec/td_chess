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
            agent_name = 'worker_' + str(task_index)
            agent = TDLeafAgent(agent_name,
                                network,
                                env,
                                verbose=True)
            global_episode_count = tf.train.get_or_create_global_step()
            increment_global_episode_count_op = global_episode_count.assign_add(1)
            summary_op = tf.summary.merge_all()

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0 and job_name == 'worker'),
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=100,
                                               scaffold=tf.train.Scaffold(summary_op=summary_op)) as sess:
            agent.sess = sess

            if job_name == "worker":
                while not sess.should_stop():
                    sess.run(increment_global_episode_count_op)
                    episode_count = sess.run(agent.global_episode_count)
                    if (episode_count - 1) % 2000 < 14:
                        agent.test((episode_count - 1) % 1000, depth=1)
                    else:
                        agent.train(depth=1)


if __name__ == "__main__":
    ps_hosts = ['localhost:' + str(2222 + i) for i in range(6)]
    worker_hosts = ['localhost:' + str(3333 + i) for i in range(48)]
    ckpt_dir = "./log/" + str(int(time.time()))
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    processes = []

    for task_idx, _ in enumerate(ps_hosts):
        p = Process(target=work, args=(None, 'ps', task_idx, cluster_spec, ckpt_dir,))
        processes.append(p)
        p.start()
        time.sleep(1)

    for task_idx, _ in enumerate(worker_hosts):
        env = ChessEnv(load_pgn=True, load_tests=True)
        # env = TicTacToeEnv()
        p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir,))
        processes.append(p)
        p.start()
        time.sleep(.5)

    for process in processes:
        process.join()
