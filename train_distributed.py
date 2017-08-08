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

            fv_size = env.get_feature_vector_size()

            if job_name == "tester":
                # network = ValueModel(fv_size)
                network = ChessValueModel()
                agent_name = 'tester_' + str(task_index)
                agent = TDLeafAgent(agent_name,
                                    network,
                                    env,
                                    verbose=True)
            else:
                # network = ValueModel(fv_size)
                network = ChessValueModel()
                agent_name = 'trainer_' + str(task_index)
                agent = TDLeafAgent(agent_name,
                                    network,
                                    env,
                                    verbose=False)

            summary_op = tf.summary.merge_all()

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0 and job_name == 'trainer'),
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=10,
                                               scaffold=tf.train.Scaffold(summary_op=summary_op)) as mon_sess:
            agent.sess = mon_sess

            if job_name == "trainer":
                while not mon_sess.should_stop():
                    agent.train(depth=1)

            elif job_name == "tester":
                while not mon_sess.should_stop():
                    agent.test(task_idx, depth=1)
                    # # TODO: distribute tests among testers
                    # for test_idx in range(14):
                    #     # agent.random_agent_test(depth=3)
                    #     agent.test(test_idx, depth=3)


if __name__ == "__main__":
    ps_hosts = ['localhost:2222']
    tester_hosts = ['localhost:' + str(3333 + i) for i in range(1)]
    trainer_hosts = ['localhost:' + str(4444 + i) for i in range(2)]
    ckpt_dir = "./log/" + str(int(time.time()))
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "tester": tester_hosts, "trainer": trainer_hosts})

    processes = []

    for task_idx, _ in enumerate(ps_hosts):
        p = Process(target=work, args=(None, 'ps', task_idx, cluster, ckpt_dir,))
        processes.append(p)
        p.start()

    for task_idx, _ in enumerate(tester_hosts):
        env = ChessEnv()
        # env = TicTacToeEnv()
        p = Process(target=work, args=(env, 'tester', task_idx, cluster, ckpt_dir,))
        processes.append(p)
        p.start()

    for task_idx, _ in enumerate(trainer_hosts):
        env = ChessEnv()
        # env = TicTacToeEnv()
        p = Process(target=work, args=(env, 'trainer', task_idx, cluster, ckpt_dir,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
