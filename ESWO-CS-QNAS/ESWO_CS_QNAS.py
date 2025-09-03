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
"""python ESWO-CS-QNAS.py"""
import argparse
import time
import os
import numpy as np
from src.weak_predictor import weak_model_train
import src.utils.logger as lg
from src.utils.config import cfg
from src.dataset import create_loaders
from src.eswo import eswo
from src.utils.train_utils import train_qnn_final
from src.model.common import create_qnn

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    train_start = time.time()
    parser = argparse.ArgumentParser(description='eswo-cs-qnas parser')
    parser.add_argument('--data-type', type=str, default="mnist", help='mnist dataset type')
    parser.add_argument('--data-path', type=str, default="./dataset/mnist/", help='mnist dataset path')
    parser.add_argument('--batch', type=int, default=32, help='train batch size')
    parser.add_argument('--epoch', type=int, default=3, help='train epoch')
    parser.add_argument('--final', type=int, default=10, help='final train epoch')
    parser.add_argument('--candidate', type=int, default=1, help='the number of candidates per envolutionary code')
    parser.add_argument('--noise', type=int, default=0, help='add noise')
    args = parser.parse_args()
    cfg.DATASET.type = args.data_type
    cfg.DATASET.path = args.data_path
    cfg.TRAIN.checkpoint_path = "./weights/" + cfg.DATASET.type + "/final/"
    cfg.TRAIN.BATCH_SIZE = args.batch
    cfg.TRAIN.EPOCHS = args.epoch  # 10 for warship
    cfg.TRAIN.EPOCHS_FINAL = args.final  # 20 for warship
    cfg.LOG_NAME = "train_" + cfg.DATASET.type
    cfg.QEA.candidate = args.candidate
    if cfg.DATASET.type == "thucnews":
      cfg.QEA.Genome = 96
      cfg.QEA.genomeLength = cfg.QEA.Genome + 1
      cfg.TRAIN.learning_rate = 0.01
      cfg.QEA.N = 3  # Initial Population size
      cfg.QEA.N_min = 3
      cfg.QEA.popSize = cfg.QEA.N + 1
      cfg.QEA.qpv = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength])
      cfg.QEA.chromosome = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength], dtype=np.int64)
      cfg.QEA.generation_max = 2
      cfg.QEA.best_acc = np.empty([cfg.QEA.generation_max])  # Save the best accuracy of each generation
      cfg.QEA.best_arch = np.empty([cfg.QEA.generation_max, cfg.QEA.genomeLength])
      cfg.QEA.fitness = np.zeros([cfg.QEA.popSize])
      cfg.QEA.best_chrom = np.empty([cfg.QEA.generation_max])
    cfg.QEA.candidate_qpv=np.zeros([cfg.QEA.candidate,cfg.QEA.genomeLength])
    cfg.QEA.candidate_chrom=np.zeros([cfg.QEA.candidate,cfg.QEA.genomeLength])
    cfg.TRAIN.noise = args.noise
    cfg.ROOT = os.path.dirname(__file__)
    os.chdir(cfg.ROOT)

    logger = lg.get_logger(cfg.LOG_NAME)
    create_loaders(cfg)
    if cfg.QEA.candidate > 1:
      weak_model_train()
    eswo()
    logger.info("-------------------------final-------------")
    logger.info("best_acc: %s", str(cfg.QEA.best_acc))
    logger.info("best_arch: %s", str(cfg.QEA.best_arch))
    best_acc_list = cfg.QEA.best_acc.tolist()
    acc_set = set()
    acc_set.add(round(max(best_acc_list),4))
    max_index = best_acc_list.index(max(best_acc_list))
    logger.info('best model is gen_indiv[%s][%s]', str(max_index), str(cfg.QEA.best_chrom[max_index]))
    best_accuracy = train_qnn_final(cfg.QEA.best_arch[max_index], max_index, cfg.QEA.best_chrom[max_index],
                                    cfg.TRAIN.EPOCHS_FINAL)
    best_gen = max_index
    best_indiv = cfg.QEA.best_chrom[max_index]
    
    best_arch = cfg.QEA.best_arch[max_index]                              
    extra_test = int(args.epoch*(290-cfg.QEA.train_times)/args.final)#the number of extra evaluation
    key_value = {}
    for i in range(1,cfg.QEA.N_min+1):
      key_value[round(cfg.QEA.fitness[i],4)]=i
    t=0
    logger.info('cfg.QEA.Fitness_total: %s :', str(cfg.QEA.fitness[1:cfg.QEA.N_min+1]))
    
    for i in sorted(key_value,reverse=True):
      if i in acc_set:
        continue
      else:
        logger.info('cfg.QEA.Fitness: %s indiv_num:%s:', i,key_value[i])
        acc = train_qnn_final(cfg.QEA.chromosome[key_value[i]], cfg.QEA.generation_max-1, key_value[i],
                                    cfg.TRAIN.EPOCHS_FINAL)
        if best_accuracy<acc:
            best_gen = cfg.QEA.generation_max-1
            best_indiv = key_value[i]
            best_accuracy=acc
            best_arch=cfg.QEA.chromosome[key_value[i]]
        acc_set.add(i)
        t=t+1
        if t>=extra_test:
          break
    if not os.path.exists(cfg.TRAIN.checkpoint_path):
        os.system("mkdir -p " + cfg.TRAIN.checkpoint_path)
    os.system("cp -r ./weights/final/gen_{}/ind_{}acc_{}%/* ".format(best_gen, best_indiv, np.round(best_accuracy, 10)
                                                                    * 100) + cfg.TRAIN.checkpoint_path)
    
    logger.info('best accuracy : %s with quantum circuit :', str(best_accuracy))
    logger.info(create_qnn(best_arch))
    complete = time.time() - train_start
    lg_info = 'Training complete in ({:.0f}h {:.0f}m {:.0f}s)'.format(complete // 3600, complete // 60, complete % 60)
    logger.info(lg_info)
    total_epochs = cfg.TRAIN.EPOCHS * cfg.QEA.N * cfg.QEA.generation_max + cfg.TRAIN.EPOCHS_FINAL
    steps = cfg.TRAIN.DATA_SIZE * total_epochs
    speed = (complete * 1000) / steps
    lg_info = "{:.0f}ms/step on average".format(speed)
    logger.info(lg_info)