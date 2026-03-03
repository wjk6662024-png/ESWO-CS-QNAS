# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
The implementation of quantum evolutionary algorithm
"""
import logging
import math
import os
import glob
import numpy as np
from src.weak_predictor import weak_filter
from src.utils.config import cfg
from src.utils.train_utils import train_qnn

def init_population(p_alpha):
    cfg.QEA.qpv=2*np.random.rand(cfg.QEA.popSize,cfg.QEA.genomeLength)-1
    for i in range(1, cfg.QEA.popSize):
        for j in range(cfg.QEA.genomeLength):
            if p_alpha <= cfg.QEA.qpv[i, j]:
                cfg.QEA.chromosome[i, j]= 1
            else:
                cfg.QEA.chromosome[i, j]= -1

# Obverse
def measure(p_alpha,i):
    for j in range(cfg.QEA.genomeLength):
        if p_alpha <= cfg.QEA.qpv[i, j]:
            cfg.QEA.chromosome[i, j] = 1
        else:
            cfg.QEA.chromosome[i, j] = -1
def measure_candidate(p_alpha):
    for i in range(cfg.QEA.candidate):
      for j in range(cfg.QEA.genomeLength):
        if p_alpha <= cfg.QEA.candidate_qpv[i, j]:
            cfg.QEA.candidate_chrom[i, j] = 1
        else:
            cfg.QEA.candidate_chrom[i, j] = -1
def fitness_evaluation(generation,record_fitness):#不需要传种群大小，因为这个函数仅在第0代会调用
    fitness_total = 0
    sum_sqr = 0
    # fitness_average = 0
    # variance = 0
    logger = logging.getLogger(cfg.LOG_NAME)
    for i in range(1,cfg.QEA.popSize):
        #cfg.QEA.fitness[i] = 0
        # Constructing quantum neural network based on chromosome
        qnn_results,msg = train_qnn(cfg.QEA.chromosome[i], generation, i, cfg.TRAIN.EPOCHS)
        #qnn_results=1
        #msg=1
        record_fitness[bytes(cfg.QEA.chromosome[i])]=[qnn_results,msg]
        logger.info("start_chrom%send_chrom",str(cfg.QEA.chromosome[i]))
        logger.info("start_fitness%send_fitness",str(qnn_results))
        cfg.QEA.fitness[i]=qnn_results
        # logger.info("fitness = ",f," ",fitness[i])
        fitness_total = fitness_total + cfg.QEA.fitness[i]
    fitness_average = fitness_total / cfg.QEA.N
    cfg.QEA.av_fitness.append(fitness_average)
    i = 1
    while i <= cfg.QEA.N:
        sum_sqr = sum_sqr + pow(cfg.QEA.fitness[i] - fitness_average, 2)
        i = i + 1
    variance = sum_sqr / cfg.QEA.N
    if variance <= 1.0e-4:
        variance = 0.0
    # Best chromosome selection
    the_best_chrom = 1
    fitness_max = cfg.QEA.fitness[1]
    for i in range(1,cfg.QEA.popSize):
        if cfg.QEA.fitness[i] >= fitness_max:
            fitness_max = cfg.QEA.fitness[i]
            the_best_chrom = i
    cfg.QEA.best_chrom[generation] = the_best_chrom
    cfg.QEA.fitness_best.append(cfg.QEA.fitness[the_best_chrom])
    cfg.QEA.best_acc[generation] = cfg.QEA.fitness[the_best_chrom]
    cfg.QEA.best_arch[generation] = cfg.QEA.chromosome[the_best_chrom]
    logger = logging.getLogger(cfg.LOG_NAME)
    logger.info("Population size = %s", str(cfg.QEA.popSize - 1))
    logger.info("mean fitness = %s", str(fitness_average))
    lg_info = "variance = " + str(variance) + " Std. deviation = " + str(math.sqrt(variance))
    logger.info(lg_info)
    logger.info("fitness max = %s", str(cfg.QEA.fitness[the_best_chrom]))
def fitness_eval(generation,i,record_fitness):
    #is_add=True#判断是否评估了一个个体
    if bytes(cfg.QEA.chromosome[i]) in record_fitness:
        [qnn_results,msg]=record_fitness[bytes(cfg.QEA.chromosome[i])]
        if qnn_results>=cfg.QEA.fitness[i]:
          print("第{}代第{}个个体与之前个体重复，无需训练".format(generation,i))
          #将msg对应的评估文件复制到对应文件位置
          dir_init = msg
          os.system("mkdir -p ./weights/gen_{}/ind_{}acc_{}%".format(generation, i, np.round(qnn_results, 10) * 100))
          os.system("cp "+msg+"/model.arch "+msg+"/init.ckpt "+msg+"/best.ckpt "+msg+"/latest.ckpt "
              "./weights/gen_{}/ind_{}acc_{}%".format(generation, int(i), np.round(qnn_results, 10) * 100))
        else:
          os.system("mkdir -p ./weights/gen_{}/ind_{}acc_{}%".format(generation, i, np.round(cfg.QEA.fitness[i], 10) * 100))
          dir_init = "./weights/gen_{}/ind_{}*".format(generation-1, int(i))
          if len(glob.glob(dir_init)) == 1:
            dir_init = glob.glob(dir_init)[0]
          else:
            dir_init = glob.glob(dir_init)[-1]
          os.system("cp "+dir_init+"/model.arch "+dir_init+"/init.ckpt "+dir_init+"/best.ckpt "+dir_init+"/latest.ckpt "
              "./weights/gen_{}/ind_{}acc_{}%".format(generation, int(i), np.round(cfg.QEA.fitness[i], 10) * 100))
        #is_add=False
    else:
        #qnn_results=1
        #msg=1
        qnn_results,msg = train_qnn(cfg.QEA.chromosome[i], generation, i, cfg.TRAIN.EPOCHS)
        record_fitness[bytes(cfg.QEA.chromosome[i])]=[qnn_results,msg]
        #is_add=True
    logger = logging.getLogger(cfg.LOG_NAME)
    logger.info("start_chrom%send_chrom",str(cfg.QEA.chromosome[i]))
    logger.info("start_fitness%send_fitness",str(qnn_results))
    return qnn_results
def generation_summary(generation,SearchAgents_no):
    fitness_total = 0
    sum_sqr = 0
    # fitness_average = 0
    # variance = 0
    logger = logging.getLogger(cfg.LOG_NAME)
    for i in range(1,SearchAgents_no+1):
        fitness_total = fitness_total + cfg.QEA.fitness[i]
    fitness_average = fitness_total / SearchAgents_no
    cfg.QEA.av_fitness.append(fitness_average)
    i = 1
    while i <= SearchAgents_no:
        sum_sqr = sum_sqr + pow(cfg.QEA.fitness[i] - fitness_average, 2)
        i = i + 1
    variance = sum_sqr / SearchAgents_no
    if variance <= 1.0e-4:
        variance = 0.0
    # Best chromosome selection
    the_best_chrom = 1
    fitness_max = cfg.QEA.fitness[1]
    for i in range(1,SearchAgents_no+1):
        if cfg.QEA.fitness[i] >= fitness_max:
            fitness_max = cfg.QEA.fitness[i]
            the_best_chrom = i
    cfg.QEA.best_chrom[generation] = the_best_chrom
    cfg.QEA.fitness_best.append(cfg.QEA.fitness[the_best_chrom])
    cfg.QEA.best_acc[generation] = cfg.QEA.fitness[the_best_chrom]
    cfg.QEA.best_arch[generation] = cfg.QEA.chromosome[the_best_chrom]
    logger = logging.getLogger(cfg.LOG_NAME)
    logger.info("Population size = %s", str(SearchAgents_no))
    logger.info("mean fitness = %s", str(fitness_average))
    lg_info = "variance = " + str(variance) + " Std. deviation = " + str(math.sqrt(variance))
    logger.info(lg_info)
    logger.info("fitness max = %s", str(cfg.QEA.fitness[the_best_chrom]))
def Levy(d):
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(np.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=np.random.normal(1,d)*sigma
    v=np.random.normal(1,d)
    step=u/abs(v)**(1/beta)
    return 0.05*step
def range_limit(wasp,lb,ub):
    for j in range(cfg.QEA.genomeLength):
        if wasp[j]<lb:
            wasp[j]=-1
        if wasp[j]>ub:
            wasp[j]=1
# qea
def eswo():
    generation = 0
    logger = logging.getLogger(cfg.LOG_NAME)
    record_fitness={}
    lg_info = "============== GENERATION: " + str(generation) + " =========================== \n"
    logger.info(lg_info)
    SearchAgents_no=cfg.QEA.N
    init_population(0)
    hungry_sense=np.zeros(cfg.QEA.popSize)
    hungry=np.zeros(cfg.QEA.popSize)
    fitness_evaluation(generation,record_fitness)  
    Best_SW=cfg.QEA.best_arch[generation].copy()
    fitness_best=cfg.QEA.fitness_best[0]
    while generation < cfg.QEA.generation_max-1:
        generation = generation + 1
        SearchAgents_no=round(cfg.QEA.N-(cfg.QEA.N-cfg.QEA.N_min+1)*generation/cfg.QEA.generation_max)
        if SearchAgents_no<cfg.QEA.N_min:
          SearchAgents_no=cfg.QEA.N_min
        lg_info = "============== GENERATION: " + str(generation) + " =========================== \n"
        logger.info(lg_info)
        worst_fitness=min(cfg.QEA.fitness)
        lg_info = "The best arch index of generation [" + str(generation) + "] is " + \
                  str(cfg.QEA.best_chrom[generation]) + '\n'
        logger.info(lg_info)
        k=(1-generation/cfg.QEA.generation_max)
        a=2*k
        a2=-1-(generation/cfg.QEA.generation_max)
        JK=np.random.permutation(SearchAgents_no)+1
        
        if np.random.random()<cfg.QEA.TR:
            sum_hungry=0
            for i in range(1,SearchAgents_no+1):
                hungry_sense[i]=(cfg.QEA.fitness[i]-np.max(cfg.QEA.fitness_best))*np.random.random()*2/(worst_fitness-np.max(cfg.QEA.fitness_best))
                if cfg.QEA.fitness[i]==np.max(cfg.QEA.fitness_best):
                    hungry[i]=0
                else:
                    hungry[i]=hungry[i]+hungry_sense[i]
            sum_hungry=sum(hungry)
            for i in range(1,SearchAgents_no+1):
                #r1=np.random.random()
                #r2=np.random.random()
                #r3=np.random.random()
                #p=np.random.random()
                #C=a*(2*r1-1)
                #l=(a2-1)*np.random.random()+1#3*np.random.random()-2
                #L=Levy(1)
                #vc = np.random.uniform(low=-k,high=k,size=cfg.QEA.genomeLength)
                #rn1=np.random.normal()
                O_P=cfg.QEA.qpv[i].copy()
                O_chromosome=cfg.QEA.chromosome[i].copy()
                #searching and nesting
                for t_t in range(cfg.QEA.candidate):
                  p=np.random.random()
                  JK=np.random.permutation(SearchAgents_no)+1
                  rn1=np.random.normal()
                  r1=np.random.random()
                  r2=np.random.random()
                  r3=np.random.random()
                  C=a*(2*r1-1)
                  vc = np.random.uniform(low=-k,high=k,size=cfg.QEA.genomeLength)
                  l=(a2-1)*np.random.random()+1#3*np.random.random()-2
                  L=Levy(1)
                  if i<k*SearchAgents_no:
                   if p<k:
                    if r1<r2:
                        m1=abs(rn1)*r1
                        cfg.QEA.candidate_qpv[t_t]=cfg.QEA.qpv[i]+m1*(cfg.QEA.qpv[JK[0]]-cfg.QEA.qpv[JK[1]])
                    else:
                        r_ao=np.random.randint(1,21)+0.00565*np.random.randint(1,cfg.QEA.genomeLength)
                        cfg.QEA.candidate_qpv[t_t]=Best_SW*Levy(cfg.QEA.genomeLength)+cfg.QEA.qpv[JK[i]]+(r_ao*math.cos(-0.005*cfg.QEA.Genome+math.pi*3/2)-r_ao*math.sin(-0.005*cfg.QEA.Genome+math.pi*3/2))*np.random.random()
                   else:
                    if r1<r2:
                        cfg.QEA.candidate_qpv[t_t]=cfg.QEA.qpv[i]+C*abs(2*np.random.rand(cfg.QEA.genomeLength)*cfg.QEA.qpv[JK[2]]-cfg.QEA.qpv[i])
                    else:
                        cfg.QEA.candidate_qpv[t_t]=cfg.QEA.qpv[i]*vc
                  else:
                    if r1<r2:
                        cfg.QEA.candidate_qpv[t_t]=Best_SW+math.cos(2*l*math.pi)*(Best_SW-cfg.QEA.qpv[i])*(2*np.random.random()*(1-np.exp(-abs(hungry[i]-sum_hungry))))
                    else:
                        cfg.QEA.candidate_qpv[t_t]=cfg.QEA.qpv[JK[0]]+r3*abs(L)*(cfg.QEA.qpv[JK[0]]-cfg.QEA.qpv[i])+(1-r3)*(np.random.rand(cfg.QEA.genomeLength)>np.random.rand(cfg.QEA.genomeLength))*(cfg.QEA.qpv[JK[2]]-cfg.QEA.qpv[JK[1]])
                measure_candidate(0)
                weak_filter(i)
                range_limit(cfg.QEA.qpv[i], -1, 1)
                qnn_results=fitness_eval(generation,i,record_fitness)
                if qnn_results>=cfg.QEA.fitness[i]:
                    cfg.QEA.fitness[i]=qnn_results
                    if qnn_results>=fitness_best:
                        fitness_best=qnn_results
                        Best_SW=cfg.QEA.chromosome[i].copy()
                else:
                    logger.info("未更新")
                    cfg.QEA.qpv[i]=O_P.copy()
                    cfg.QEA.chromosome[i]=O_chromosome.copy()
                
                logger.info("%s %s %s",str(generation),str(i),str(cfg.QEA.chromosome[i]))
       #Mating behavior交叉
        else:    
            for i in range(1,SearchAgents_no+1):
                #l=(a2-1)*np.random.random()+1
                #SW_m=np.zeros(cfg.QEA.genomeLength)
                O_P=cfg.QEA.qpv[i].copy()
                O_chromosome=cfg.QEA.chromosome[i].copy()
                #if cfg.QEA.fitness[JK[0]]>cfg.QEA.fitness[i]:
                #    v1=cfg.QEA.qpv[JK[0]]-cfg.QEA.qpv[i]
                #else:
                #    v1=cfg.QEA.qpv[i]-cfg.QEA.qpv[JK[0]]
                #if cfg.QEA.fitness[JK[1]]>cfg.QEA.fitness[JK[2]]:
                #    v2=cfg.QEA.qpv[JK[1]]-cfg.QEA.qpv[JK[2]]
                #else:
                #    v2=cfg.QEA.qpv[JK[2]]-cfg.QEA.qpv[JK[1]]
                #rn1=np.random.normal()
                #rn2=np.random.normal()
                for t_t in range(cfg.QEA.candidate):
                  JK=np.random.permutation(SearchAgents_no)+1
                  l=(a2-1)*np.random.random()+1
                  SW_m=np.zeros(cfg.QEA.genomeLength)
                  if cfg.QEA.fitness[JK[0]]>cfg.QEA.fitness[i]:
                    v1=cfg.QEA.qpv[JK[0]]-cfg.QEA.qpv[i]
                  else:
                    v1=cfg.QEA.qpv[i]-cfg.QEA.qpv[JK[0]]
                  if cfg.QEA.fitness[JK[1]]>cfg.QEA.fitness[JK[2]]:
                    v2=cfg.QEA.qpv[JK[1]]-cfg.QEA.qpv[JK[2]]
                  else:
                    v2=cfg.QEA.qpv[JK[2]]-cfg.QEA.qpv[JK[1]]
                  rn1=np.random.normal()
                  rn2=np.random.normal()
                  cfg.QEA.candidate_qpv[t_t]=cfg.QEA.qpv[i]
                  for j in range(1,cfg.QEA.genomeLength):
                    SW_m[j]=cfg.QEA.qpv[i,j]+((np.exp(l))*abs(rn1)*v1[j]+(1-np.exp(l))*abs(rn2)*v2[j])
                    if np.random.random()<cfg.QEA.CR:
                        cfg.QEA.candidate_qpv[t_t,j]=round(SW_m[j], 2)
                measure_candidate(0)
                weak_filter(i)
                range_limit(cfg.QEA.qpv[i], -1, 1)
                qnn_results=fitness_eval(generation,i,record_fitness)
                if qnn_results>=cfg.QEA.fitness[i]:
                    cfg.QEA.fitness[i]=qnn_results
                    if qnn_results>=fitness_best:
                        fitness_best=qnn_results
                        Best_SW=cfg.QEA.chromosome[i].copy()#数组整体赋值时要万分小心
                else:
                    logger.info("未更新")
                    cfg.QEA.qpv[i]=O_P.copy()
                    cfg.QEA.chromosome[i]=O_chromosome.copy()
                logger.info("%s %s %s %s",str(generation),str(i),str(7),str(cfg.QEA.qpv[i]))
        generation_summary(generation,SearchAgents_no)
        logger.info(cfg.QEA.train_times)