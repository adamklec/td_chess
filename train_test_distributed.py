from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from envs.tic_tac_toe import TicTacToeEnv
from multiprocessing import Process
import time
import tensorflow as tf
from value_model import ValueModel
from chess_value_model import ChessValueModel


def work(env, job_name, task_index, cluster, log_dir, verbose, random_agent_test):

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

            test_period = 1000
            num_tests = 14
            while not sess.should_stop():
                episode_number = sess.run(agent.increment_train_episode_count)
                # test_idx = (episode_number-1) % test_period
                # if test_idx < num_tests:
                #     if random_agent_test:
                #         result = agent.random_agent_test(depth=3)
                #         for update_op, result in zip(agent.update_random_agent_test_results, result):
                #             agent.sess.run(update_op, feed_dict={agent.test_result_: result})
                #
                #         global_episode_count = sess.run(agent.update_count)
                #
                #         if verbose:
                #             print("EPISODE:", global_episode_count, "RANDOM AGENT TEST")
                #             print('FIRST PLAYER:',
                #                   sess.run([agent.first_player_wins,
                #                             agent.first_player_draws,
                #                             agent.first_player_losses]))
                #             print('SECOND PLAYER:',
                #                   sess.run([agent.second_player_wins,
                #                             agent.second_player_draws,
                #                             agent.second_player_losses]))
                #             print('-' * 100)
                #     else:
                #         result = agent.test(test_idx, depth=3)
                #         sess.run(agent.update_test_results, feed_dict={agent.test_idx_: test_idx,
                #                                                        agent.test_result_: result})
                #         if agent.verbose:
                #             test_results = agent.sess.run(agent.test_results)
                #             print(worker_name,
                #                   "EPISODE:", episode_number,
                #                   "UPDATE:", sess.run(agent.update_count),
                #                   "TEST INDEX:", test_idx,
                #                   "RESULT:", result)
                #             print(test_results, "\n", "TOTAL:", sum(test_results))
                #             print('-' * 100)

                # else:
                reward = agent.train(num_moves=10, depth=3)
                if agent.verbose:
                    print(worker_name,
                          "EPISODE:", episode_number,
                          "UPDATE:", sess.run(agent.update_count),
                          "REWARD:", reward)
                    print('-' * 100)


if __name__ == "__main__":
    ps_hosts = ['54.84.212.199:' + str(2222 + i) for i in range(5)]
    worker_hosts = ['54.84.212.199:' + str(3333 + i) for i in range(40)]
    ckpt_dir = "./log/" + str(int(time.time()))
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    processes = []

    for task_idx, _ in enumerate(ps_hosts):
        p = Process(target=work, args=(None, 'ps', task_idx, cluster_spec, ckpt_dir, 1, False))
        processes.append(p)
        p.start()
        time.sleep(1)

    for task_idx, _ in enumerate(worker_hosts):
        env = ChessEnv()
        # env = TicTacToeEnv()
        p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir, 1, False))
        processes.append(p)
        p.start()
        time.sleep(1)

    for process in processes:
        process.join()
