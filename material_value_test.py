from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from multiprocessing import Process
import tensorflow as tf
from value_model import ValueModel
from os import listdir
from os.path import isfile, join
import re
import time


def parse_test_string(string):
    data = [s for s in string.split('; ')]
    d = dict()
    d['fen'] = data[0].split(' bm ')[0] + " 0 0"
    d['bm'] = data[0].split(' bm ')[1]

    for c in data[1:]:
        c = c.replace('"', '')
        c = c.replace(';', '')
        item = c.split(maxsplit=1, sep=" ")
        d[item[0]] = item[1]

    move_rewards = {}
    answers = d['c0'].split(',')
    for answer in answers:
        move_reward = answer.split('=')
        move_rewards[move_reward[0].strip()] = int(move_reward[1])
    d['c0'] = move_rewards
    return d


def work(job_name, task_index, cluster, log_dir, verbose):

    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)
    if job_name == "ps":
        server.join()
    else:
        env = ChessEnv()
        worker_device = "/job:tester/task:%d" % task_index
        with tf.device(tf.train.replica_device_setter(
                worker_device=worker_device,
                cluster=cluster)):

            with tf.device(worker_device):
                with tf.variable_scope('local'):
                    local_network = ValueModel(is_local=True)

            network = ValueModel()

            worker_name = 'worker_%03d' % task_index
            agent = TDLeafAgent(worker_name,
                                network,
                                local_network,
                                env,
                                verbose=verbose)

            test_path = "./chess_tests/"
            test_filenames = sorted([f for f in listdir(test_path) if isfile(join(test_path, f))])
            test_strings = []
            for filename in test_filenames:
                with open(test_path + filename) as f:
                    for string in f:
                        test_strings.append(string.strip())

            summary_op = tf.summary.merge_all()
            scaffold = tf.train.Scaffold(summary_op=summary_op)
        is_chief = (task_idx == 0)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=1,
                                               scaffold=scaffold) as sess:
            agent.sess = sess

            num_tests = 1400
            episode_number = 0
            while episode_number < 2000:
                episode_number = sess.run(agent.increment_test_episode_count)
                test_idx = (episode_number-1) % num_tests
                d = parse_test_string(test_strings[test_idx])
                result = agent.test(d, depth=1, use_material_value=True)

                filename = test_filenames[test_idx]
                matches = re.split('-|\.', filename)
                row_idx = int(matches[0])
                test_idx = int(matches[1][-2:]) - 1

                sess.run(agent.update_test_results, feed_dict={agent.test_idx_: test_idx,
                                                               agent.row_idx_: row_idx,
                                                               agent.test_result_: result})
                if agent.verbose:
                    test_results_reduced, elo_estimate = agent.sess.run([agent.test_results_reduced, agent.elo_estimate])
                    print(worker_name,
                          "EPISODE:", episode_number,
                          "UPDATE:", sess.run(agent.update_count),
                          "TEST INDEX:", test_idx,
                          "FILENAME:", filename,
                          "RESULT:", result)
                    print(test_results_reduced, "\n", "TOTAL:", sum(test_results_reduced))
                    print("ELO ESTIMATE:", elo_estimate)
                    print('-' * 100)


if __name__ == "__main__":
    ps_hosts = ['localhost:' + str(2222 + i) for i in range(1)]
    tester_hosts = ['localhost:' + str(3333 + i) for i in range(5)]

    ckpt_dir = "./log/test_" + str(time.time())
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ps_hosts,
         "tester": tester_hosts})

    processes = []
    for task_idx, _ in enumerate(ps_hosts):
        p = Process(target=work, args=('ps', task_idx, cluster_spec, ckpt_dir, 1))
        processes.append(p)
        p.start()

    for task_idx, _ in enumerate(tester_hosts):
        p = Process(target=work, args=('tester', task_idx, cluster_spec, ckpt_dir, 1))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
