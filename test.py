import tensorflow as tf

from agents.nn_agent import NeuralNetworkAgent
from game import Chess
import pandas as pd


def parse_tests(fn):
    with open(fn, "r") as f:
        tests = f.readlines()

    dicts = []
    data = [[s for s in test.split('; ')] for test in tests[:3]]
    for row in data:
        d = {}
        d['fen'] = row[0]
        for c in row[1:]:
            c = c.replace('"', '')
            c = c.replace(';\n', '')
            item = c.split(maxsplit=1, sep=" ")
            d[item[0]] = item[1]
        dicts.append(d)

    for d in dicts:
        move_rewards = {}
        answers = d['c0'].split(',')
        for answer in answers:
            move_reward = answer.split('=')
            move_rewards[move_reward[0].strip()] = int(move_reward[1])
        d['c0'] = move_rewards
    df = pd.DataFrame.from_dict(dicts)
    df = df.set_index('id')
    return df

def main():
    filename = "/Users/adam/Documents/projects/td_chess/STS[1-13]/STS4.epd"
    parse_tests(filename)

    with tf.Session() as sess:
        nn_agent = NeuralNetworkAgent(sess)
        env = Chess()
        nn_agent.train(env, 10000, 2, 0.1, verbose=True)

if __name__ == "__main__":
    main()
