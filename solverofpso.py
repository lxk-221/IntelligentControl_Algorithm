import argparse
import random
from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt

def readdatafile():
    pathodfile = f'newdata.txt'
    with open(pathodfile) as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines]
    n_m = lines[0]
    n, m = list(map(int, n_m.split()))
    sch = []
    tis = {}
    for i in range(n):
        tis[i] = {}
        for j in range(m):
            tis[i][j] = None
    for i in range(n):
        line = list(map(float, lines[i + 1].split()))
        index, p = 0, []
        while index < len(line):
            machine = line[index]
            p.append(int(machine))
            index += 1
            time = line[index]
            index += 1
            tis[i][int(machine)] = time
        sch.append(p)
    return n, m, tis, sch


def timecalculate(n, m, item, times, schedule):
    # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
    processed_id = [0] * n
    machineWorkTime = [0] * m
    
    # 全部工件的每到工序的开始结束时间
    startTime = [[0 for _ in range(m)] for _ in range(n)]
    endTime = [[0 for _ in range(m)] for _ in range(n)]
    
    final_time = 0
    for wId in item:
        # 依据粒子信息，得到当前的需考虑的工件id
        # 依据当前工件的工序得到处理的机器以及耗时
        pId = processed_id[wId]
        processed_id[wId] += 1
        mId = schedule[wId][pId]
        t = times[wId][mId]
        if pId == 0:
            startTime[wId][pId] = machineWorkTime[mId]
        else:
            startTime[wId][pId] = max(endTime[wId][pId - 1], machineWorkTime[mId])
        machineWorkTime[mId] = startTime[wId][pId] + t
        endTime[wId][pId] = machineWorkTime[mId]
        final_time = max(final_time, machineWorkTime[mId])
    return final_time


def particlesinit(n, m, size=100):
    particles = []
    init_seq = []
    for i in range(n):
        init_seq.extend([i] * m)
    for _ in range(size):
        seq = deepcopy(init_seq)
        random.shuffle(seq)
        while seq in particles:
            random.shuffle(seq)
        particles.append(seq)
    return particles


def getswitchpairs(x, y, n):
    def findidxitem(seq, item, nidx):
        counter = 0
        for idx, i in enumerate(seq):
            if i == item:
                counter += 1
            if counter == nidx:
                return idx

    current, l, pairs = 0, len(x), []
    y_counter = [1] * n
    while current < l:
        # 最优粒子当前的考虑的工件编号
        item_x = x[current]
        # 当前粒子，最优粒子对应的工件序号，该工件对应的工序号
        idx_y = findidxitem(y, item_x, y_counter[item_x])
        y_counter[item_x] += 1
        if idx_y != current:
            pairs.append((current, idx_y))
        current += 1
        print("pairs:", pairs)
    return pairs


def pairinit(length, size):
    rest = []
    while len(rest) < size:
        pair = np.random.choice(length, 2).tolist()
        if pair[0] != pair[1]:
            rest.append(pair)
    return rest


def applypairs(x, pairs):
    for p_x, p_y in pairs:
        x[p_x], x[p_y] = x[p_y], x[p_x]


class solverofpso:
    def __init__(self, n, m, times, schedule):
        self.args = {'ep': 500, 'pnum': 100, 'n': n, 'm': m,
                     'spairs': 5, 'alpha': 0.45, 'maxpairslen': 10, 'se': 100, }
        self.pbest = [inf] * self.args['pnum']
        self.gbest = inf
        self.particles = particlesinit(
            self.args['n'], self.args['m'], self.args['pnum'])
        self.vecs = [pairinit(self.args['n'] * self.args['m'], self.args['spairs']) \
                     for _ in range(self.args['pnum'])]
        self.pbest_solution = deepcopy(self.particles)
        self.gbest_solution = None
        self.times = times
        self.schedule = schedule

    def start(self):
        random.seed(self.args['se'])
        np.random.seed(self.args['se'])
        fit_plt = []
        for e in range(self.args['ep']):
            # 更新粒子自身最优
            for idx, p in enumerate(self.particles):
                f = timecalculate(
                    self.args['n'], self.args['m'], p,
                    self.times, self.schedule)
                if f < self.pbest[idx]:
                    self.pbest[idx] = f
                    self.pbest_solution[idx] = deepcopy(p)
            gbest_idx, gbest_rest = 0, inf
            # 更新粒子群全体最优
            for idx in range(len(self.particles)):
                if self.pbest[idx] < gbest_rest:
                    gbest_idx = idx
                    gbest_rest = self.pbest[idx]
            self.gbest = gbest_rest
            fit_plt.append(self.gbest)
            print("self.gbest ", self.gbest)
            self.gbest_solution = deepcopy(self.pbest_solution[gbest_idx])

            # 粒子的编号
            for idx in range(len(self.particles)):
                p = self.particles[idx]
                gbest_delta = getswitchpairs(self.gbest_solution, p, self.args['n'])
                pbest_delta = getswitchpairs(self.pbest_solution[idx], p, self.args['n'])
                pairs = []
                if random.random() < self.args['alpha']:
                    pairs = pbest_delta
                else:
                    pairs = gbest_delta
                vec = self.vecs[idx] + pairs
                print("vec ", vec)
                if len(vec) > self.args['maxpairslen']:
                    vec = random.sample(vec, self.args['maxpairslen'])
                    self.vecs[idx] = vec
                applypairs(self.particles[idx], vec)

        print(f'best result: {self.gbest}')
        plt.plot(np.arange(len(fit_plt)), fit_plt)
        plt.show()


if __name__ == "__main__":
    parr = argparse.ArgumentParser()
    parr.add_argument('--item', default=0, type=int)
    args = parr.parse_args()
    n, m, times, schedule = readdatafile()
    finalout = solverofpso(n, m, times, schedule)
    finalout.start()
