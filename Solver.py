import argparse
import random
from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt

class SubSolver:
    def __init__(self):
        pass

    ## 特定算法自身数据的初始化
    def init(self):
        print("your SubSolver need a init function!")

    ## 迭代
    def update(self):
        print("your SubSolver need a update function!")

    ## 一次实验
    def loop(self):
        print("your SubSolver need a loop function!")

    ## 绘图
    def plot(self):
        print("your SubSolver need a plot function to show your results!")


###GA算法求解###
class SolverGA(SubSolver):
    def __init__(self, WorkPiece_num, Process_num, TimeCost_matrix, MachineRequired_matrix):
        # 数据参数，来自文件
        self.WorkPiece_num = WorkPiece_num
        self.Process_num = Process_num
        self.TimeCost_matrix = TimeCost_matrix
        self.MachineRequired_matrix = MachineRequired_matrix

        # 算法可调节参数, 变异率，交叉率
        self.rateofmutation = 0.1
        self.rateofcrossover = 0.65

        # 初始化参数
        self.init()

    ## 特定算法自身数据的初始化
    def init(self):
        # 初始化粒子群
        self.Particles = self.particlesInit(self.WorkPiece_num, self.Process_num, self.args['Particle_num'])
        # 随机速度
        self.RandomVelocity = [self.randomPairInit(self.WorkPiece_num * self.Process_num, self.args['RandomPair_num']) for _ in range(self.args['Particle_num'])]
        
        # 初始化粒子最优与粒子群最优，及其对应的解
        self.pbest = [inf] * self.args['Particle_num']
        self.gbest = inf
        self.pbest_solution = self.Particles
        self.gbest_solution = self.Particles[0]

        # 用于绘制曲线图
        self.CostHistory = []

    ## 迭代
    def update(self):
        print("your SubSolver need a update function!")

    ## 一次实验
    def loop(self):
        print("your SubSolver need a loop function!")

    ## 绘图
    def plot(self):
        print("your SubSolver need a plot function to show your results!")


###PSO算法求解###
class SolverPSO(SubSolver):
    def __init__(self, WorkPiece_num, Process_num, TimeCost_matrix, MachineRequired_matrix):
        # 数据参数，来自文件
        self.WorkPiece_num = WorkPiece_num
        self.Process_num = Process_num
        self.TimeCost_matrix = TimeCost_matrix
        self.MachineRequired_matrix = MachineRequired_matrix

        # 算法可调节参数，迭代次数、粒子群粒子数量、随机交换对的数量、学习率、最大交换数量、随机种子（保证可重复性）
        self.args = {'Iteration': 500, 'Particle_num': 100, 'RandomPair_num': 5, 'alpha': 0.45, 'MaxPair_num': 10, 'RandomSeed': 100, }

        # 初始化参数
        self.init()

    ##
    def init(self):
        # 初始化粒子群
        self.Particles = self.particlesInit(self.WorkPiece_num, self.Process_num, self.args['Particle_num'])
        # 随机速度
        self.RandomVelocity = [self.randomPairInit(self.WorkPiece_num * self.Process_num, self.args['RandomPair_num']) for _ in range(self.args['Particle_num'])]
        
        # 初始化粒子最优与粒子群最优，及其对应的解
        self.pbest = [inf] * self.args['Particle_num']
        self.gbest = inf
        self.pbest_solution = self.Particles
        self.gbest_solution = self.Particles[0]

        # 用于绘制曲线图
        self.CostHistory = []

    def particlesInit(self, workpiece_num, process_num, particle_num=100):
        particles = []
        init_seq = []
        
        # 单个粒子，也就是原问题的一个解
        for workpiece_index in range(workpiece_num):
            init_seq.extend([workpiece_index] * process_num)

        for _ in range(particle_num):
            seq = deepcopy(init_seq)
            random.shuffle(seq)
            # 保证当前粒子不在粒子群中，在的话重新打乱，避免相同粒子出现
            while seq in particles:
                random.shuffle(seq)
            particles.append(seq)

        return particles

    def randomPairInit(self, Particle_length, RandomPair_num):
        RandomPairs = []
        # 生成一定数量的交换对，可以理解为粒子的随机速度
        while len(RandomPairs) < RandomPair_num:
            tmp_pair = np.random.choice(Particle_length, 2).tolist()
            # 检查是否为有效交换
            if tmp_pair[0] != tmp_pair[1]:
                RandomPairs.append(tmp_pair)
        return RandomPairs

    def update(self):
        for particle_index, particle in enumerate(self.Particles):
            # 根据粒子给出的工件顺序，计算消耗的总用时
            time_cost = self.timeCalculate(self.WorkPiece_num,
                                   self.Process_num, 
                                   particle,
                                   self.TimeCost_matrix, 
                                   self.MachineRequired_matrix)
            # 更新粒子自身最优记录
            if time_cost < self.pbest[particle_index]:
                self.pbest[particle_index] = time_cost
                self.pbest_solution[particle_index] = deepcopy(particle)
            
        # 更新粒子群的最优记录，从各个粒子最优记录中找到最优的
        gbest_idx, gbest = 0, inf
        for particle_index in range(self.args['Particle_num']):
            if self.pbest[particle_index] < gbest:
                gbest_idx = particle_index
                gbest = self.pbest[particle_index]
        self.gbest = gbest
        self.gbest_solution = self.pbest_solution[gbest_idx]

        # 储存并输出
        self.CostHistory.append(self.gbest)
        print("self.gbest", self.gbest)

        # 更新粒子群
        for particle_index in range(self.args['Particle_num']):
            particle = self.Particles[particle_index]
            gbest_delta = self.getswitchpairs(self.gbest_solution, particle, self.WorkPiece_num)
            pbest_delta = self.getswitchpairs(self.pbest_solution[particle_index], particle, self.WorkPiece_num)
            pairs = []
            if random.random() < self.args['alpha']:
                pairs = pbest_delta
            else:
                pairs = gbest_delta
            vec = self.RandomVelocity[particle_index] + pairs
            
            if len(vec) > self.args['MaxPair_num']:
                vec = random.sample(vec, self.args['MaxPair_num'])
                self.RandomVelocity[particle_index] = vec
            
            self.applypairs(self.Particles[particle_index], vec)
    
    def getswitchpairs(self, x, y, n):
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
        return pairs
    
    def timeCalculate(self, workpiece_num, machine_num, item, times, schedule):
        # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
        processed_id = [0] * workpiece_num
        machineWorkTime = [0] * machine_num
        
        # 全部工件的每到工序的开始结束时间
        startTime = [[0 for _ in range(machine_num)] for _ in range(workpiece_num)]
        endTime = [[0 for _ in range(machine_num)] for _ in range(workpiece_num)]
        
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
    
    def applypairs(self, x, pairs):
        for p_x, p_y in pairs:
            x[p_x], x[p_y] = x[p_y], x[p_x]

    def loop(self):
        # 设定随机种子，保证实验可重复性
        #random.seed(self.args['RandomSeed'])
        #np.random.seed(self.args['RandomSeed'])

        for iter in range(self.args['Iteration']):
            print("Iteration ", iter, ":")
            self.update()

    def plot(self):
        plt.plot(np.arange(len(self.CostHistory)), self.CostHistory)
        plt.show()


class Solver:
    def __init__(self):
        self.WorkPiece_num = 0
        self.Process_num = 0
        self.TimeCost_matrix = None
        self.MachineRequired_matrix = None

    def readFile(self, filename):
        # 读取文件内容并去掉全0行
        with open(filename) as f:
            lines = f.readlines()
            lines = [i.strip() for i in lines]
        
        # 机器数与工序数
        param = lines[0]
        self.WorkPiece_num, self.Process_num = list(map(int, param.split()))

        sch = []
        tis = {}

        for i in range(self.WorkPiece_num):
            tis[i] = {}
            for j in range(self.Process_num):
                tis[i][j] = None

        print(tis)
        for i in range(self.WorkPiece_num):
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

        self.TimeCost_matrix = tis
        self.MachineRequired_matrix = sch

    def setSolvers(self, solvers):
        if(solvers == 'PSO'):
            self.SubSolver = SolverPSO(self.WorkPiece_num, self.Process_num, self.TimeCost_matrix, self.MachineRequired_matrix)
        else:
            print("Solver should be GA\PSO\ACA!")

    def solve(self):
        self.SubSolver.loop()
        self.SubSolver.plot()
        
if __name__ == "__main__":
    solver = Solver()
    solver.readFile('newdata.txt')
    solver.setSolvers('PSO')
    solver.solve()


