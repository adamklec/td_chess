from agents.nn_agent import NeuralNetworkAgent
from game import Chess
from multiprocessing import Process
import time


def work(job_name, task_index, ps_hosts, worker_hosts):
    import tensorflow as tf
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)

    if job_name == "ps" or job_name == 'master':
        server.join()

    elif job_name == "worker":
        print(job_name, task_index, 'target:', server.target)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            global_episode_count = tf.contrib.framework.get_or_create_global_step()
            agent = NeuralNetworkAgent('agent_' + str(task_index), global_episode_count, verbose=True)

        hooks = [tf.train.StopAtStepHook(last_step=1000)]

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               checkpoint_dir="log/"+str(int(time.time())),
                                               hooks=hooks) as mon_sess:
            print(job_name, task_index, 'created mon_sess')

            while not mon_sess.should_stop():
                print(job_name, task_index, 'training')
                agent.train(mon_sess, Chess(), 100, 0.05, pretrain=False)
                # print(agent.test_function(mon_sess))
                # print(task_index, 'result: ', mon_sess.run(x))


if __name__ == "__main__":
    ps_hosts = ['localhost:2222', 'localhost:2223']
    worker_hosts = ['localhost:2224', 'localhost:2225']

    processes = []

    for task_idx, ps_host in enumerate(ps_hosts):
        print(ps_host, task_idx)
        p = Process(target=work, args=('ps', task_idx, ps_hosts, worker_hosts))
        processes.append(p)
        p.start()

    print('ps started')

    for task_idx, worker_host in enumerate(worker_hosts):
        print(worker_host, task_idx)
        p = Process(target=work, args=('worker', task_idx, ps_hosts, worker_hosts,))
        processes.append(p)
        p.start()
        time.sleep(2)

    for process in processes:
        process.join()
