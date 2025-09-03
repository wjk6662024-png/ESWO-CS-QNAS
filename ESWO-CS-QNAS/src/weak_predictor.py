import json
import os
import numpy as np
from src.utils.config import cfg

def weak_model_train():
    current_directory = os.getcwd()
    if cfg.DATASET.type=="mnist" or cfg.DATASET.type=="thucnews":
      if cfg.DATASET.type=="mnist":
        file_name = "src/dataset/weak_train_data/mnist.txt" # 200
      elif cfg.DATASET.type=="thucnews":
        file_name = "src/dataset/weak_train_data/thucnews.txt" # 98
      file_path = os.path.join(current_directory, file_name)
      with open(file_path, 'r') as f:
        json_data = json.load(f)
        X_train=np.array(json_data[0])
        y_train=np.array(json_data[1])
      X_train[X_train==0]=-1
      cfg.model.rf.fit(X_train, y_train)
      cfg.model.gnb.fit(X_train, y_train)
      cfg.model.knn.fit(X_train, y_train)
      cfg.model.svmc.fit(X_train, y_train)
    elif cfg.DATASET.type=="warship":
      file_name = "src/dataset/weak_train_data/warship.txt" # 200
      file_path = os.path.join(current_directory, file_name)
      X_train_number_list=[]
      with open(file_path, 'r') as f:
        json_data = json.load(f)
        X_train=np.array(json_data[0])
        y_train=np.array(json_data[1])
      X_train[X_train==0]=-1
      cfg.model.rf.fit(X_train, y_train)
      cfg.model.bc.fit(X_train, y_train)
      cfg.model.svmc.fit(X_train, y_train)
def weak_filter(indiv):#choose the best from ten individuals
    score=np.zeros(cfg.QEA.candidate)#begining from the number one
    print(cfg.QEA.candidate)
    if cfg.QEA.candidate==1:
      top1=0
    elif cfg.QEA.candidate>1:  
      if cfg.DATASET.type=="mnist" or cfg.DATASET.type=="thucnews":
        for i in range(cfg.QEA.candidate):
          for j in range(cfg.QEA.candidate):
            if i!=j:
              tmp=np.concatenate((cfg.QEA.candidate_chrom[i][1:],cfg.QEA.candidate_chrom[j][1:])).reshape(1,-1)
              y1=cfg.model.knn.predict(tmp)
              y2=cfg.model.rf.predict(tmp)
              y3=cfg.model.gnb.predict(tmp)
              y4=cfg.model.svmc.predict(tmp)
              y_sum=2*y1+y2+y3+y4
              y_sum=y_sum.sum()
              if y_sum>2:#representing the former is better
                score[i]=score[i]+1
              else:
                score[j]=score[j]+1
      elif cfg.DATASET.type=="warship":
        for i in range(cfg.QEA.candidate):
          for j in range(cfg.QEA.candidate):
            if i!=j:
              tmp=np.concatenate((cfg.QEA.candidate_chrom[i][1:],cfg.QEA.candidate_chrom[j][1:])).reshape(1,-1)
              y1=cfg.model.bc.predict(tmp)
              y2=cfg.model.rf.predict(tmp)
              y3=cfg.model.svmc.predict(tmp)
              y_sum=y1+y2+y3
              y_sum=y_sum.sum()
              if y_sum>=2:#representing the former is better
                score[i]=score[i]+1
              else:
                score[j]=score[j]+1
      #find top1
      print("score",score)
      top1=np.argmax(score)
    for j in range(1,cfg.QEA.genomeLength):
      cfg.QEA.chromosome[indiv][j]=cfg.QEA.candidate_chrom[top1][j]
      cfg.QEA.qpv[indiv][j]=round(cfg.QEA.candidate_qpv[top1][j],2)
    #Dealing with situations where the ansatz is empty
    if np.sum(cfg.QEA.chromosome[indiv][1:])==cfg.QEA.Genome:
        cfg.QEA.qpv[indiv]=2*np.random.rand(cfg.QEA.genomeLength)-1
        for j in range(1, cfg.QEA.genomeLength):
            if cfg.QEA.qpv[indiv, j] >= 0:
                cfg.QEA.chromosome[indiv, j]= 1
            else:
                cfg.QEA.chromosome[indiv, j]= -1