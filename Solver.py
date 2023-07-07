import argparse
import random
from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt

class SubSolver:
    def __init__(self, WorkPiece_num, Process_num, TimeCost_matrix, MachineRequired_matrix, RandomSeed = None):
        # 数据参数，来自文件
        self.WorkPiece_num = WorkPiece_num
        self.Process_num = Process_num
        self.Machine_num = 10
        self.TimeCost_matrix = TimeCost_matrix
        self.MachineRequired_matrix = MachineRequired_matrix

        self.RandomSeed = RandomSeed

        # 记录最优解, 用于绘制甘特图
        self.best_cost = inf
        self.best_solution = None

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
    def __init__(self, WorkPiece_num, Process_num, TimeCost_matrix, MachineRequired_matrix, RandomSeed = None):
        # 数据参数，来自文件
        self.WorkPiece_num = WorkPiece_num
        self.Process_num = Process_num
        self.Machine_num = 10
        self.TimeCost_matrix = TimeCost_matrix
        self.MachineRequired_matrix = MachineRequired_matrix

        self.RandomSeed = RandomSeed

        # 记录最优解, 用于绘制甘特图
        self.best_cost = inf
        self.best_solution = None

        # 设定随机种子，保证实验可重复性
        if(self.RandomSeed != None):
            random.seed(self.RandomSeed)
            np.random.seed(self.RandomSeed)

        # 算法可调节参数, 变异率，交叉率
        self.Iteration = 1000
        self.Population_num = 100
        self.MutationRate = 0.1
        self.CrossoverRate = 0.65
              
    ## 特定算法自身数据的初始化
    def init(self):
        # 染色体长度
        self.Chromosome_length = self.WorkPiece_num * self.Process_num

        # 初始化种群及适应度
        self.Fitness = np.zeros(self.Population_num)
        self.Population = self.populationInit()
        

        # 用于绘制曲线图
        self.CostHistory = []

    def populationInit(self):
        # 种群
        Population = np.zeros((self.Population_num, self.Chromosome_length), dtype = int)
        for individual in range(self.Population_num):
            # 基因包括不同工件号，如1出现1次代表1的第一道工序，1出现2次代表1的第二道工序
            for workpiece_index in range(self.WorkPiece_num):
                for process in range(self.Process_num):
                    Population[individual][workpiece_index * self.Process_num + process] = workpiece_index
            # 打乱
            np.random.shuffle(Population[individual][:self.Chromosome_length])
            # 计算适应度
            self.Fitness[individual] = self.timeCalculate(Population[individual])
        return Population

    def timeCalculate(self, gene):
            # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
            processed_id = [0] * self.WorkPiece_num
            machineWorkTime = [0] * self.Machine_num
            
            # 全部工件的每到工序的开始结束时间
            startTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            endTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            
            final_time = 0
            
            for wId in gene:
                # 依据基因信息，得到当前的需考虑的工件id
                # 依据当前工件的工序得到处理的机器以及耗时
                pId = processed_id[wId]
                processed_id[wId] += 1
                mId = self.MachineRequired_matrix[wId][pId]
                t = self.TimeCost_matrix[wId][mId]
                if pId == 0:
                    startTime[wId][pId] = machineWorkTime[mId]
                else:
                    startTime[wId][pId] = max(endTime[wId][pId - 1], machineWorkTime[mId])
                machineWorkTime[mId] = startTime[wId][pId] + t
                endTime[wId][pId] = machineWorkTime[mId]
                final_time = max(final_time, machineWorkTime[mId])
            return final_time
    
    def crossoverAndMutation(self):
        NextPopulations = np.zeros((self.Population_num, self.Chromosome_length), dtype = int)
        # 对每一个个体
        for individual in range(self.Population_num):
            # 自身作为第一个父代
            father1 = self.Population[individual]
            # 交叉
            if np.random.random() < self.CrossoverRate:
                # 随机选取另一个个体作为第二个父代
                father2 = self.Population[np.random.randint(self.Population_num)]
                
                seq = [j for j in range(self.WorkPiece_num)]
                random_length = np.random.randint(2, len(seq) - 1)
                set1 = set()

                # 取random_length个工件到set1里
                for _ in range(random_length):
                    index = np.random.randint(0, len(seq))
                    set1.add(seq[index])
                    seq.pop(index)

                # 其余工件放到set2里
                set2 = set(seq)

                # 生成两个子代
                child1 = np.copy(father1)
                child2 = np.copy(father2)
                
                # 找到set2中有的工件在两个子代中的位序
                remain1 = [i for i in range(self.Chromosome_length) if father1[i] in set2]
                remain2 = [i for i in range(self.Chromosome_length) if father2[i] in set2]
                
                cursor1, cursor2 = 0, 0
                for k in range(self.Chromosome_length):
                    if father2[k] in set2:
                        child1[remain1[cursor1]] = father2[k]
                        cursor1 += 1
                    if father1[k] in set2:
                        child2[remain2[cursor2]] = father1[k]
                        cursor2 += 1
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                if self.timeCalculate(child1) < self.timeCalculate(child2):
                    child = child1
                else:
                    child = child2
            else:
                child = father1
            NextPopulations[individual] = child
        return NextPopulations

    def mutation(self, child):
        # 如果变异，就任意交换两位
        if np.random.rand() < self.MutationRate:
            index = np.random.randint(self.Chromosome_length, size=2)
            child[index[0]], child[index[1]] = child[index[1]], child[index[0]]
        return child

    def choseNextPopulation(self, NewPopulation):
        # 根据适应度选择了种群个数个基因
        # 选择较好的抗体
        
        reserve_num = int(self.Population_num/100)

        # 旧抗体的匹配度
        temp_fitness = self.Fitness
        # 新抗体的匹配度
        for i in range(len(NewPopulation)):
            temp_fitness = np.append(temp_fitness, self.timeCalculate(NewPopulation[i]))
        
        sort_index = np.argsort(temp_fitness)
        
        # 最终选择的抗体的下标集合
        chossed_index = []
        for temp_index in sort_index[:reserve_num]:
            chossed_index.append(temp_index)

        fitness_sum = np.sum(1/temp_fitness)
        idx = np.random.choice(np.arange(self.Population_num + len(NewPopulation)),
                               size=self.Population_num - reserve_num,
                               replace=True,
                               p=1 / temp_fitness / fitness_sum)
        chossed_index = np.concatenate((chossed_index, idx)) 

        save_Population = []
        save_fitness = []

        for new_population_index in range(self.Population_num):
            if(chossed_index[new_population_index] < self.Population_num):
                save_Population.append(self.Population[chossed_index[new_population_index]])
            else:
                save_Population.append(NewPopulation[chossed_index[new_population_index]-self.Population_num])
            save_fitness.append(temp_fitness[chossed_index[new_population_index]])
        
        self.Population = save_Population
        self.Fitness = save_fitness

    ## 迭代
    def update(self):
        new_population = self.crossoverAndMutation()
        self.choseNextPopulation(new_population)
        self.CostHistory.append(min(self.Fitness))
        if(np.min(self.Fitness) < self.best_cost):
            self.best_solution = self.Population[np.argmin(self.Fitness)]
            self.best_cost = np.min(self.Fitness)

    ## 一次实验
    def loop(self):
        
        # 初始化参数
        self.init()

        for iter in range(self.Iteration):
            print("Iteration", iter, ":", min(self.Fitness))
            self.update()

        if(self.RandomSeed != None):
            print("RandomSeed is:", self.RandomSeed)
            print("Best solution is:", min(self.CostHistory))

        print("self.best_solution:", self.best_solution)    

    ## 绘图
    def plot(self):
        plt.plot(np.arange(len(self.CostHistory)), self.CostHistory)
        plt.show()
        
        self.getGanttData(self.best_solution)
        fig, ax = plt.subplots()
        # 不同工件的颜色
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink','tab:gray', 'tab:olive', 'tab:cyan']
        ax.set_xlim(0, 1300)
        y_para1 = []
        y_para2 = []
        for machine_index in range(self.Machine_num):
            
            y_para1.append(15*(machine_index + 1))
            y_para2.append('Machine' + str(machine_index))
            
            x_para1 = []
            x_para2 = (15*(machine_index + 1) - 5, 10)
            x_para3 = []
            # time_interval 是增量的形式
            for time_interval, work_piece_id in self.WorkOnMachines[machine_index]:
                print("time_interval:",time_interval)
                x_para1.append(time_interval)
                x_para3.append(color[work_piece_id])
                #ax.broken_barh([time_interval], (15*(machine_index + 1) - 5, 10), facecolors=(color[work_piece_id]),
                #               edgecolor='black', linewidth=1)

            x_para3 = tuple(x_para3)
            print("x_para1:", x_para1)
            print("x_para2:", x_para2)
            print("x_para3:", x_para3)
            ax.broken_barh(x_para1, x_para2, facecolors=x_para3,
                           edgecolor='black', linewidth=1)
        #print("self.WorkOnMachines:", self.WorkOnMachines)
        ax.set_yticks(y_para1, labels=y_para2)
        
        # 更改y轴记号标签
        #ax.set_yticks([15, 25], labels=['A01', 'A02'])
        
        # 设置条形数据
        #ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')
        
        #ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors=(color[0], color[1], color[2]))
        plt.show()

    def getGanttData(self, one_solution):
        # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
            self.WorkOnMachines = [[]] * self.Machine_num
            

            processed_id = [0] * self.WorkPiece_num
            machineWorkTime = [0] * self.Machine_num
            
            # 全部工件的每到工序的开始结束时间
            startTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            endTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            
            final_time = 0
            
            for wId in one_solution:
                # 依据信息，得到当前的需考虑的工件id
                # 依据当前工件的工序得到处理的机器以及耗时
                pId = processed_id[wId]
                processed_id[wId] += 1
                mId = self.MachineRequired_matrix[wId][pId]
                t = self.TimeCost_matrix[wId][mId]
                if pId == 0:
                    startTime[wId][pId] = machineWorkTime[mId]
                else:
                    startTime[wId][pId] = max(endTime[wId][pId - 1], machineWorkTime[mId])
                machineWorkTime[mId] = startTime[wId][pId] + t
                endTime[wId][pId] = machineWorkTime[mId]
                final_time = max(final_time, machineWorkTime[mId])

                # 如果某一机器还没添加时间节点及对应工件号，则初始化
                if(self.WorkOnMachines[mId] == []):
                    self.WorkOnMachines[mId] = [ ((startTime[wId][pId],endTime[wId][pId]-startTime[wId][pId]), wId) ]
                else:
                    self.WorkOnMachines[mId].append( ((startTime[wId][pId],endTime[wId][pId]-startTime[wId][pId]), wId) )

###PSO算法求解###
class SolverPSO(SubSolver):
    def __init__(self, WorkPiece_num, Process_num, TimeCost_matrix, MachineRequired_matrix, RandomSeed = None):
        # 数据参数，来自文件
        self.WorkPiece_num = WorkPiece_num
        self.Process_num = Process_num
        self.Machine_num = 10
        self.TimeCost_matrix = TimeCost_matrix
        self.MachineRequired_matrix = MachineRequired_matrix

        self.RandomSeed = RandomSeed

        # 记录最优解, 用于绘制甘特图
        self.best_cost = inf
        self.best_solution = None

        # 设定随机种子，保证实验可重复性
        if(self.RandomSeed != None):
            random.seed(self.RandomSeed)
            np.random.seed(self.RandomSeed)

        # 算法可调节参数，迭代次数、粒子群粒子数量、随机交换对的数量、学习率、最大交换数量、随机种子（保证可重复性）
        self.args = {'Iteration': 500, 'Particle_num': 100, 'RandomPair_num': 5, 'alpha': 0.45, 'MaxPair_num': 10, 'RandomSeed': 100, }

        # 初始化参数
        self.init()

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
        if(gbest < self.best_cost):
            self.best_solution = self.Particles[gbest_idx]
            self.best_cost = gbest

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
        for iter in range(self.args['Iteration']):
            print("Iteration", iter, ":", self.best_cost)
            self.update()

    ## 绘图
    def plot(self):
        plt.plot(np.arange(len(self.CostHistory)), self.CostHistory)
        plt.show()

        self.getGanttData(self.gbest_solution)
        fig, ax = plt.subplots()
        # 不同工件的颜色
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink','tab:gray', 'tab:olive', 'tab:cyan']
        ax.set_xlim(0, 1300)
        y_para1 = []
        y_para2 = []
        for machine_index in range(self.Machine_num):
            
            y_para1.append(15*(machine_index + 1))
            y_para2.append('Machine' + str(machine_index))
            
            x_para1 = []
            x_para2 = (15*(machine_index + 1) - 5, 10)
            x_para3 = []
            # time_interval 是增量的形式
            for time_interval, work_piece_id in self.WorkOnMachines[machine_index]:
                #print("time_interval:",time_interval)
                x_para1.append(time_interval)
                x_para3.append(color[work_piece_id])
                #ax.broken_barh([time_interval], (15*(machine_index + 1) - 5, 10), facecolors=(color[work_piece_id]),
                #               edgecolor='black', linewidth=1)

            x_para3 = tuple(x_para3)
            #print("x_para1:", x_para1)
            #print("x_para2:", x_para2)
            #print("x_para3:", x_para3)
            ax.broken_barh(x_para1, x_para2, facecolors=x_para3,
                           edgecolor='black', linewidth=1)
        #print("self.WorkOnMachines:", self.WorkOnMachines)
        ax.set_yticks(y_para1, labels=y_para2)
        
        # 更改y轴记号标签
        #ax.set_yticks([15, 25], labels=['A01', 'A02'])
        
        # 设置条形数据
        #ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')
        
        #ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors=(color[0], color[1], color[2]))
        plt.show()

    def getGanttData(self, one_solution):
        # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
            self.WorkOnMachines = [[]] * self.Machine_num
            

            processed_id = [0] * self.WorkPiece_num
            machineWorkTime = [0] * self.Machine_num
            
            # 全部工件的每到工序的开始结束时间
            startTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            endTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            
            final_time = 0
            
            for wId in one_solution:
                # 依据信息，得到当前的需考虑的工件id
                # 依据当前工件的工序得到处理的机器以及耗时
                pId = processed_id[wId]
                processed_id[wId] += 1
                mId = self.MachineRequired_matrix[wId][pId]
                t = self.TimeCost_matrix[wId][mId]
                if pId == 0:
                    startTime[wId][pId] = machineWorkTime[mId]
                else:
                    startTime[wId][pId] = max(endTime[wId][pId - 1], machineWorkTime[mId])
                machineWorkTime[mId] = startTime[wId][pId] + t
                endTime[wId][pId] = machineWorkTime[mId]
                final_time = max(final_time, machineWorkTime[mId])

                # 如果某一机器还没添加时间节点及对应工件号，则初始化
                if(self.WorkOnMachines[mId] == []):
                    self.WorkOnMachines[mId] = [ ((startTime[wId][pId],endTime[wId][pId]-startTime[wId][pId]), wId) ]
                else:
                    self.WorkOnMachines[mId].append( ((startTime[wId][pId],endTime[wId][pId]-startTime[wId][pId]), wId) )

###AIA算法求解###
class SolverAIA(SubSolver):
    def __init__(self, WorkPiece_num, Process_num, TimeCost_matrix, MachineRequired_matrix, RandomSeed = None):
        # 数据参数，来自文件
        self.WorkPiece_num = WorkPiece_num
        self.Process_num = Process_num
        self.Machine_num = 10
        self.TimeCost_matrix = TimeCost_matrix
        self.MachineRequired_matrix = MachineRequired_matrix

        self.RandomSeed = RandomSeed
        
        # 记录最优解, 用于绘制甘特图
        self.best_cost = inf
        self.best_solution = None

        # 设定随机种子，保证实验可重复性
        if(self.RandomSeed != None):
            random.seed(self.RandomSeed)
            np.random.seed(self.RandomSeed)

        # 算法可调节参数，迭代次数，抗体数量，换位率，移位率，逆转率，重组率
        self.Iteration = 1000
        
        self.Population_num = 100
        
        self.SwitchRate = 0.8
        self.ShiftRate = 0.8
        self.InverseRate = 0.8
        self.RegroupRate = 0.8
        
        
    ## 特定算法自身数据的初始化
    def init(self):
        # 抗体长度
        self.Antibody_length = self.WorkPiece_num * self.Process_num

        # 初始化种群及适应度
        self.Fitness = np.zeros(self.Population_num)
        self.Population = self.populationInit()
        
        # 用于绘制曲线图
        self.CostHistory = []

    def populationInit(self):
        # 抗体种群
        Population = np.zeros((self.Population_num, self.Antibody_length), dtype = int)
        for individual in range(self.Population_num):
            # 抗体包括不同工件号，如1出现1次代表1的第一道工序，1出现2次代表1的第二道工序
            for workpiece_index in range(self.WorkPiece_num):
                for process in range(self.Process_num):
                    Population[individual][workpiece_index * self.Process_num + process] = workpiece_index
            # 打乱
            np.random.shuffle(Population[individual][:self.Antibody_length])
            # 计算适应度
            self.Fitness[individual] = self.timeCalculate(Population[individual])
        return Population

    def timeCalculate(self, gene):
            # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
            processed_id = [0] * self.WorkPiece_num
            machineWorkTime = [0] * self.Machine_num
            
            # 全部工件的每到工序的开始结束时间
            startTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            endTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            
            final_time = 0
            
            for wId in gene:
                # 依据抗体信息，得到当前的需考虑的工件id
                # 依据当前工件的工序得到处理的机器以及耗时
                pId = processed_id[wId]
                processed_id[wId] += 1
                mId = self.MachineRequired_matrix[wId][pId]
                t = self.TimeCost_matrix[wId][mId]
                if pId == 0:
                    startTime[wId][pId] = machineWorkTime[mId]
                else:
                    startTime[wId][pId] = max(endTime[wId][pId - 1], machineWorkTime[mId])
                machineWorkTime[mId] = startTime[wId][pId] + t
                endTime[wId][pId] = machineWorkTime[mId]
                final_time = max(final_time, machineWorkTime[mId])
            return final_time
    
    def generateNewAntiBodys(self):
        NewAntiBodys = []

        # 对每一个抗体
        for antibody in self.Population:
            antibody = self.switch(antibody)
            antibody = self.shift(antibody)
            antibody = self.inverse(antibody)
            antibody = self.regroup(antibody)
            
            NewAntiBodys.append(antibody)

        return NewAntiBodys

    def switch(self, antibody):
        # 换位
        if np.random.rand() < self.SwitchRate:
            # 变化多少对，最多选取int(self.Antibody_length/4)对
            switch_num = np.random.randint(int(self.Antibody_length/4), size=1)[0]
            index = np.random.randint(self.Antibody_length, size=2*switch_num)
            
            # 对每一对进行交换
            for i in range(switch_num):
                antibody[index[2*i]], antibody[index[2*i+1]] = antibody[index[2*i+1]], antibody[index[2*i]]

        return antibody

    def shift(self, antibody):
        # 移位多少段，最多选取int(self.Antibody_length/10)对
        shift_num = np.random.randint(int(self.Antibody_length/10), size=1)[0]
        index = np.random.randint(self.Antibody_length, size=2*shift_num)
        if np.random.rand() < self.ShiftRate:
            # 对每一段进行移位
            for i in range(shift_num):
                if index[2*i] == index[2*i+1]:
                    continue
                
                if index[2*i] < index[2*i+1]:
                    smaller_index = index[2*i]
                    bigger_index = index[2*i+1]
                else:
                    smaller_index = index[2*i+1]
                    bigger_index = index[2*i]
                # 移动一位
                antibody[smaller_index : bigger_index] = np.roll(antibody[smaller_index : bigger_index], 1)

        return antibody
    
    def inverse(self, antibody):
        # 翻转多少段，最多选取int(self.Antibody_length/10)对
        inverse_num = np.random.randint(int(self.Antibody_length/10), size=1)[0]
        index = np.random.randint(self.Antibody_length, size=2*inverse_num)
        if np.random.rand() < self.ShiftRate:
            # 对每一段进行翻转
            for i in range(inverse_num):
                if index[2*i] == index[2*i+1]:
                    continue
                
                if index[2*i] < index[2*i+1]:
                    smaller_index = index[2*i]
                    bigger_index = index[2*i+1]
                else:
                    smaller_index = index[2*i+1]
                    bigger_index = index[2*i]
                # 翻转
                antibody[smaller_index : bigger_index] = np.flip(antibody[smaller_index : bigger_index])
        
        return antibody
    
    def regroup(self, antibody):
        # 重组多少段，最多选取int(self.Antibody_length/10)对
        regroup_num = np.random.randint(int(self.Antibody_length/10), size=1)[0]
        index = np.random.randint(self.Antibody_length, size=2*regroup_num)
        if np.random.rand() < self.ShiftRate:
            # 对每一段进行重组
            for i in range(regroup_num):
                if index[2*i] == index[2*i+1]:
                    continue
                
                if index[2*i] < index[2*i+1]:
                    smaller_index = index[2*i]
                    bigger_index = index[2*i+1]
                else:
                    smaller_index = index[2*i+1]
                    bigger_index = index[2*i]
                
                # 重组
                antibody_segment = antibody[smaller_index : bigger_index]
                np.random.shuffle(antibody_segment) # shuffle返回值为None
                antibody[smaller_index : bigger_index] = antibody_segment
        
        return antibody

    def choseNextPopulation(self, NewPopulation):
        # 选择较好的抗体
        
        reserve_num = int(self.Population_num/50)

        # 旧抗体的匹配度
        temp_fitness = self.Fitness
        # 新抗体的匹配度
        for i in range(len(NewPopulation)):
            temp_fitness = np.append(temp_fitness, self.timeCalculate(NewPopulation[i]))
        
        sort_index = np.argsort(temp_fitness)
        
        # 最终选择的抗体的下标集合
        chossed_index = []
        for temp_index in sort_index[:reserve_num]:
            chossed_index.append(temp_index)

        fitness_sum = np.sum(1/temp_fitness)
        idx = np.random.choice(np.arange(self.Population_num + len(NewPopulation)),
                               size=self.Population_num - reserve_num,
                               replace=True,
                               p=1 / temp_fitness / fitness_sum)
        chossed_index = np.concatenate((chossed_index, idx)) 

        save_Population = []
        save_fitness = []

        for new_population_index in range(self.Population_num):
            if(chossed_index[new_population_index] < self.Population_num):
                save_Population.append(self.Population[chossed_index[new_population_index]])
            else:
                save_Population.append(NewPopulation[chossed_index[new_population_index]-self.Population_num])
            save_fitness.append(temp_fitness[chossed_index[new_population_index]])
        
        self.Population = save_Population
        self.Fitness = save_fitness

    ## 迭代
    def update(self):
        new_population = self.generateNewAntiBodys()
        self.choseNextPopulation(new_population)
        self.CostHistory.append(min(self.Fitness))
        if(np.min(self.Fitness) < self.best_cost):
            self.best_solution = self.Population[np.argmin(self.Fitness)]
            self.best_cost = np.min(self.Fitness)

    ## 一次实验
    def loop(self):
        
        # 初始化参数
        self.init()

        for iter in range(self.Iteration):
            print("Iteration", iter, ":", min(self.Fitness))
            self.update()

        if(self.RandomSeed != None):
            print("RandomSeed is:", self.RandomSeed)
            print("Best solution is:", min(self.CostHistory))
        
    ## 绘图
    def plot(self):
        plt.plot(np.arange(len(self.CostHistory)), self.CostHistory)
        plt.show()
        
        self.getGanttData(self.best_solution)
        fig, ax = plt.subplots()
        # 不同工件的颜色
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink','tab:gray', 'tab:olive', 'tab:cyan']
        ax.set_xlim(0, 1300)
        y_para1 = []
        y_para2 = []
        for machine_index in range(self.Machine_num):
            
            y_para1.append(15*(machine_index + 1))
            y_para2.append('Machine' + str(machine_index))
            
            x_para1 = []
            x_para2 = (15*(machine_index + 1) - 5, 10)
            x_para3 = []
            # time_interval 是增量的形式
            for time_interval, work_piece_id in self.WorkOnMachines[machine_index]:
                print("time_interval:",time_interval)
                x_para1.append(time_interval)
                x_para3.append(color[work_piece_id])
                #ax.broken_barh([time_interval], (15*(machine_index + 1) - 5, 10), facecolors=(color[work_piece_id]),
                #               edgecolor='black', linewidth=1)

            x_para3 = tuple(x_para3)
            print("x_para1:", x_para1)
            print("x_para2:", x_para2)
            print("x_para3:", x_para3)
            ax.broken_barh(x_para1, x_para2, facecolors=x_para3,
                           edgecolor='black', linewidth=1)
        #print("self.WorkOnMachines:", self.WorkOnMachines)
        ax.set_yticks(y_para1, labels=y_para2)
        
        # 更改y轴记号标签
        #ax.set_yticks([15, 25], labels=['A01', 'A02'])
        
        # 设置条形数据
        #ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')
        
        #ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors=(color[0], color[1], color[2]))
        plt.show()

    def getGanttData(self, one_solution):
        # 每个工件进行到第几道工序以及当前每个机器的结束工作时间
            self.WorkOnMachines = [[]] * self.Machine_num
            

            processed_id = [0] * self.WorkPiece_num
            machineWorkTime = [0] * self.Machine_num
            
            # 全部工件的每到工序的开始结束时间
            startTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            endTime = [[0 for _ in range(self.Machine_num)] for _ in range(self.WorkPiece_num)]
            
            final_time = 0
            
            for wId in one_solution:
                # 依据信息，得到当前的需考虑的工件id
                # 依据当前工件的工序得到处理的机器以及耗时
                pId = processed_id[wId]
                processed_id[wId] += 1
                mId = self.MachineRequired_matrix[wId][pId]
                t = self.TimeCost_matrix[wId][mId]
                if pId == 0:
                    startTime[wId][pId] = machineWorkTime[mId]
                else:
                    startTime[wId][pId] = max(endTime[wId][pId - 1], machineWorkTime[mId])
                machineWorkTime[mId] = startTime[wId][pId] + t
                endTime[wId][pId] = machineWorkTime[mId]
                final_time = max(final_time, machineWorkTime[mId])

                # 如果某一机器还没添加时间节点及对应工件号，则初始化
                if(self.WorkOnMachines[mId] == []):
                    self.WorkOnMachines[mId] = [ ((startTime[wId][pId],endTime[wId][pId]-startTime[wId][pId]), wId) ]
                else:
                    self.WorkOnMachines[mId].append( ((startTime[wId][pId],endTime[wId][pId]-startTime[wId][pId]), wId) )

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

    def setSolvers(self, solvers, RandomSeed = None):

        if(solvers == 'PSO'):
            self.SubSolver = SolverPSO(self.WorkPiece_num, self.Process_num, self.TimeCost_matrix, self.MachineRequired_matrix, RandomSeed)
        elif(solvers == 'GA'):
            self.SubSolver = SolverGA(self.WorkPiece_num, self.Process_num, self.TimeCost_matrix, self.MachineRequired_matrix, RandomSeed)
        elif(solvers == 'AIA'):
            self.SubSolver = SolverAIA(self.WorkPiece_num, self.Process_num, self.TimeCost_matrix, self.MachineRequired_matrix, RandomSeed)
        else:
            print("Solver should be GA\PSO\ACA!")

    def solve(self):
        self.SubSolver.loop()
        self.SubSolver.plot()
        
if __name__ == "__main__":
    
    use_subsolver = 'GA'

    solver = Solver()
    solver.readFile('new_data.txt')
    solver.setSolvers(use_subsolver)
    solver.solve()

    # 使用PSO
    #solver = Solver()
    #solver.readFile('data.txt')
    #solver.setSolvers('PSO')
    #solver.solve()

    # 使用GA
    #solver = Solver()
    #solver.readFile('data.txt')
    #solver.setSolvers('GA')
    #solver.solve()

    # 使用AIA
    #solver = Solver()
    #solver.readFile('data.txt')
    #solver.setSolvers('AIA')
    #solver.solve()
