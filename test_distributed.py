from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from multiprocessing import Process
import time
import tensorflow as tf
from chess_value_model import ChessValueModel
import argparse
from os import listdir
from os.path import isfile, join
import re

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

            network = ChessValueModel()

            opt = tf.train.AdamOptimizer()
            opt = tf.train.SyncReplicasOptimizer(opt, 100)

            worker_name = 'worker_%03d' % task_index
            agent = TDLeafAgent(worker_name,
                                network,
                                env,
                                opt=opt,
                                verbose=verbose)

            test_path = "./chess_tests/"
            test_filenames = sorted([f for f in listdir(test_path) if isfile(join(test_path, f))])
            test_strings = []
            for filename in test_filenames:
                with open(test_path + filename) as f:
                    for string in f:
                        test_strings.append(string.strip())

            summary_op = tf.summary.merge_all()
            # is_chief = task_index == 0
            is_chief = False
            sync_replicas_hook = opt.make_session_run_hook(is_chief)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=log_dir,
                                               save_summaries_steps=1,
                                               hooks=[sync_replicas_hook],
                                               scaffold=tf.train.Scaffold(summary_op=summary_op)) as sess:
            agent.sess = sess

            num_tests = 1400
            while not sess.should_stop():
                episode_number = sess.run(agent.increment_test_episode_count)
                test_idx = (episode_number-1) % num_tests
                d = parse_test_string(test_strings[test_idx])
                result = agent.test2(d, depth=3)
                filename = test_filenames[test_idx]
                matches = re.split('-|\.', filename)
                row_idx = int(matches[0])
                test_idx = int(matches[1][-2:]) - 1
                sess.run(agent.update_test_results, feed_dict={agent.test_idx_: test_idx,
                                                               agent.row_idx_: row_idx,
                                                               agent.test_result_: result})
                if agent.verbose:
                    test_results_reduced = agent.sess.run(agent.test_results_reduced)
                    print(worker_name,
                          "EPISODE:", episode_number,
                          "UPDATE:", sess.run(agent.update_count),
                          "TEST INDEX:", test_idx,
                          "FILENAME:", filename,
                          "RESULT:", result)
                    print(test_results_reduced, "\n", "TOTAL:", sum(test_results_reduced))
                    print('-' * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("idx")
    parser.add_argument("ips", nargs='+')
    args = parser.parse_args()

    ps_hosts = [ip + ':' + str(2222 + i) for i in range(5) for ip in args.ips]
    worker_hosts = [ip + ':' + str(3333 + i) for i in range(40) for ip in args.ips]
    ckpt_dir = "./log/" + str(int(time.time()))
    cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    this_ip = args.ips[int(args.idx)]

    processes = []

    for task_idx, ps_host in enumerate(ps_hosts):
        if this_ip in ps_host:
            p = Process(target=work, args=(None, 'ps', task_idx, cluster_spec, ckpt_dir, 2))
            processes.append(p)
            p.start()
            time.sleep(1)

    for task_idx, worker_host in enumerate(worker_hosts):
        if this_ip in worker_host:
            env = ChessEnv()
            p = Process(target=work, args=(env, 'worker', task_idx, cluster_spec, ckpt_dir, 2))
            processes.append(p)
            p.start()
            time.sleep(1)

    for process in processes:
        process.join()
