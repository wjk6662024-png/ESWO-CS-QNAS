# 目录

<!-- TOC -->

- [目录](#目录)
- [ESWO-CS-QNAS描述](#ESWO-CS-QNAS描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
- [模型描述](#模型描述)
    - [噪声](#噪声)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [每个进化节点的候选个体数量](#每个进化节点的候选个体数量)
            - [无噪声环境下在warship、mnist和thucnews数据集上训练ESWO-CS-QNAS](#无噪声环境下在warship、mnist和thucnews数据集上训练ESWO-CS-QNAS)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)
- [其他情况说明](#其他情况说明)

<!-- /TOC -->


# ESWO-CS-QNAS描述

针对基于量子电路的量子神经网络，ESWO-CS-QNAS提出了一种基于增强蛛蜂优化算法的神经网络结构搜索方法，另外还加入一种新型的缓存机制和二元代理优化机制。在本方法中，通过搜索最佳网络结构，以提高模型精度，降低量子电路的复杂性，并减轻构建实际量子电路的负担，可以用于解决分类任务，本项目解决的是图像二分类任务和文本多分类任务。

# 模型架构

该模型设计实现了一个量子神经网络用于图像二分类和文本多分类，并基于带缓存机制和二元代理优化策略的增强蛛蜂优化算法进行神经架构搜索，网络结构主要包括两个模块：

- 量子编码线路Encoder：分别使用01编码和Rx编码对不同的数据集图片以及文本向量进行编码
- 待训练线路Ansatz：使用双比特量子门（*XX*门、*YY*门、*ZZ*门）以及量子 *I* 门构建了一个两层的量子神经网络Ansatz

通过对量子神经网络输出执行泡利 z 算符测量哈密顿期望，并利用增强蛛蜂优化算法对上述量子神经网络进行架构搜索，提高模型精度，降低线路复杂度。

# 数据集

使用的数据集：

- 数据集 [MNIST](<http://yann.lecun.com/exdb/mnist/>) 描述：MNIST数据集一共有7万张图片，其中6万张是训练集，1万张是测试集。每张图片是 28×28 的0 − 9的手写数字图片组成。每个图片是黑底白字的形式，黑底用0表示，白字用0-1之间的浮点数表示，越接近1，颜色越白。本模型筛选出其中的"3"和"6"类别，进行二分类。
- 数据集 [Warship](<https://gitee.com/Pcyslist/mqnn/blob/master/warship.zip>) 描述： 为了验证QNN对更复杂图像数据集的分类效果以及我们提出的ESWO-CS-QNAS方法的有效性，我们采用了一组舰船目标数据集。该数据集是一艘航行中的船只，由无人机从不同角度拍摄。图像采用JPG格式，分辨率为640×512。它包含两个类别：Burke和Nimitz。该数据集的训练集数量为411（Burke类202个,Nimitz类209个），测试集数量为150（Burke类78个,Nimitz类72个）。
- 数据集 [THUCNews](https://pan.baidu.com/s/1hugrfRu/) 描述：THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档，均为UTF-8纯文本格式。为了降低量子神经网络的训练压力，我们从四个类别中各提取一千条数据，使用这四千条数据训练和评估搜索到的量子神经网络。
下载后，将数据集解压到如下目录：

```python
~/path/to/ESWO-CS-QNAS/src/dataset/mnist
~/path/to/ESWO-CS-QNAS/src/dataset/warship
~/path/to/ESWO-CS-QNAS/src/dataset/thucnews
```

# 环境要求

- 硬件（GPU）

    - 使用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
    - [MinQuantum](https://www.mindspore.cn/mindquantum/docs/en/r0.7/mindquantum_install.html)

- 其他第三方库安装

  ```bash
  cd ESWO-CS-QNAS
  conda create --name qnas python=3.9.13  
  conda install --name qnas --file condalist.txt
  pip install -r requirements.txt
  ```

- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindQuantum教程](https://www.mindspore.cn/mindquantum/docs/en/r0.7/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore以及MindQuantum后，您可以按照如下步骤进行训练和评估：

- GPU环境运行

  ```python
  # 训练
    # mnist 数据集训练示例
  nohup python -u ESWO_CS_QNAS.py --data-type mnist --data-path ./src/dataset/mnist/ --batch 32 --epoch 3 --final 10 --candidate 10 --noise 0 > mnist_train.log 2>&1 &
    # warship 数据集训练示例
  nohup python -u ESWO_CS_QNAS.py --data-type warship --data-path ./src/dataset/warship/ --batch 10 --epoch 15 --final 50 --candidate 10 --noise 0 > warship_train.log 2>&1 &
    # thucnews 数据集训练示例
  nohup python -u ESWO_CS_QNAS.py --data-type thucnews --data-path ./src/dataset/thucnews/ --batch 16 --epoch 3 --final 10 --candidate 10 --noise 0 > thucnews_train.log 2>&1 &

  # 训练完成之后可执行评估
  # mnist数据集评估
  python eval.py --data-type mnist --data-path ./src/dataset/mnist/ --ckpt-path /abs_path/to/best_ckpt/ | tee mnist_eval.log
  # warship数据集评估
  python eval.py --data-type warship --data-path ./src/dataset/warship/ --ckpt-path /abs_path/to/best_ckpt/ | tee warship_eval.log
  # thucnews数据集评估
  python eval.py --data-type thucnews --data-path ./src/dataset/thucnews/ --ckpt-path /abs_path/to/best_ckpt/ | tee thucnews_eval.log
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── ESWO-CS-QNAS
    ├── condalist.txt                   # Anaconda env list
    ├── ESWO_CS_QNAS.py                        # 训练脚本
    ├── eval.py                         # 评估脚本
    ├── README.md                       # ESWO-CS-QNAS模型相关说明
    ├── requirements.txt                # pip 包依赖
    └── src
        ├── dataset.py                  # 数据集生成
        ├── loss.py                     # 模型损失函数
        ├── metrics.py                  # 模型评价指标
        ├── model
        │   └── common.py               # QNN量子神经网络创建
        ├── ESWO_CS_QNAS.py             # 量子神经架构搜索算法
        └── utils
            ├── config.py               # 模型参数配置文件
            ├── data_preprocess.py      # 数据预处理
            ├── logger.py               # 日志构造器
            └── train_utils.py          # 模型训练定义
```

## 脚本参数

在config.py中可以同时配置量子进化算法参数、训练参数、数据集、和评估参数。

  ```python
  cfg = EasyDict()
  cfg.LOG_NAME = "logger"data_preprocess

  # Quantum evolution algorithm parameters
  cfg.QEA = EasyDict()
  cfg.QEA.fitness_best = []  # The best fitness of each generation

  # Various parameters of the population
  cfg.QEA.Genome = 64  # Chromosome length, 96 for thucnews
  cfg.QEA.N = 13  # initial population size
  cfg.QEA.N_min = 7 
  cfg.QEA.generation_max = 10  # Population Iterations, 30 for warship and 7 for thucnews

  # Dataset parameters
  cfg.DATASET = EasyDict()
  cfg.DATASET.type = "mnist"  # mnist or warship or thucnews
  cfg.DATASET.path = "./src/dataset/"+cfg.DATASET.type+"/"  # ./src/dataset/mnist/ or ./src/dataset/warship/ or ./src/dataset/thucnews/
  cfg.DATASET.THRESHOLD = 0.5

  # Training parameters
  cfg.TRAIN = EasyDict()
  cfg.TRAIN.EPOCHS = 3  # 15 for warship
  cfg.TRAIN.EPOCHS_FINAL = 10  # 50 for warship
  cfg.TRAIN.BATCH_SIZE = 32  # 10 for warship and 16 for thucnews
  cfg.TRAIN.learning_rate = 0.001 # 0.01 for thucnews
  cfg.TRAIN.checkpoint_path = "./weights/"+cfg.DATASET.type+"/final/"
  ```

更多配置细节请参考`utils`目录下config.py文件。

## 训练过程

### 训练

- GPU环境运行训练mnist数据集

  运行以下命令时请将数据集移动到ESWO-CS-QNAS根目录下`src/dataset`文件夹下中，则可使用相对路径描述数据集位置，否则请将`--data-path`设置为绝对路径。

  ```
  nohup python -u ESWO_CS_QNAS.py --data-type mnist --data-path ./src/dataset/mnist/ --batch 32 --epoch 3 --final 10 --candidate 10 --noise 0 > mnist_train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过当前目录下的`mnist_train.log`文件或者`./log/`目录下面的日志文件查看结果。

  训练结束后，您可在`ESWO_CS_QNAS.py`脚本所在目录下的`./weights/`目录下找到架构搜索过程中每一个模型对应的`best.ckpt、init.ckpt、latest.ckpt`文件以及`model.arch`模型架构文件。

- GPU环境运行训练warship数据集

  ```
  nohup python -u ESWO_CS_QNAS.py --data-type warship --data-path ./src/dataset/warship/ --batch 10 --epoch 15 --final 50 --candidate 10 --noise 0 > warship_train.log 2>&1 &
  ```
- GPU环境运行训练thucnews数据集

  ```
   nohup python -u ESWO_CS_QNAS.py --data-type thucnews --data-path ./src/dataset/thucnews/ --batch 16 --epoch 3 --final 10 --candidate 10 --noise 0 > thucnews_train.log 2>&1 &
  ```
  查看模型训练结果，与mnist数据集训练结果方式相同。

## 评估过程

### 评估

- 在GPU环境运行评估mnist数据集

- 在运行以下命令之前，清将数据集移动到ESWO-CS-QNAS根目录下`src/dataset`文件夹下中，则可使用相对路径描述数据集位置，否则请给出数据集的绝对路径。

- 请用于评估的检查点路径。请将检查点路径设置为绝对路径。

  ```bash
  python eval.py --data-type mnist --data-path ./src/dataset/mnist/ --ckpt-path /abs_path/to/best_ckpt/ | tee mnist_eval.log
  ```

  上述python命令将在后台运行，您可以通过mnist_eval.log文件查看结果。

- 在GPU环境运行评估warship和thucnews数据集

  请参考评估mnist数据集。

## 导出过程

### 导出MindIR

- 基于MindQuantum创建的量子模型，目前官方还不支持导出为该格式
- 但为了能够将量子线路进行保存，本项目中利用Python自带pickle数据序列化包，将架构搜索得到的每一个量子模型都保存为`./weights/model/model.arch`，您可以按照`eval.py`中的方法加载模型架构

# 模型描述
## 噪声
- noise的值为0时不添加噪声，否则所有搜索的QNN都要添加噪声，具体设置如下：所有单量子比特门添加0.1%的去极化噪声，所有双量子比特门添加1%的去极化噪声，读取比特添加4%的位翻转噪声。
## 每个进化节点的候选个体数量
- 该参数由candidate控制，candidate的值为整数且至少为1。当candidate为1时模型不使用二元代理；当candidate值大于1时，模型使用预训练获得的数据训练二元代理，在每个进化节点生成candidate个候选解并使用二元代理快速筛选获得优秀解并保留下来。

## 性能

### 训练性能

#### 无噪声环境下在warship、mnist和thucnews数据集上训练ESWO-CS-QNAS

| 参数            | GPU                                           | GPU                                           |GPU                                          |
| --------------- | --------------------------------------------- | --------------------------------------------- |--------------------------------------------- |
| 模型版本        | ESWO-CS-QNAS(noiseless)                       | ESWO-CS-QNAS(noiseless)                       |ESWO-CS-QNAS(noiseless)                       |
| 资源            | NVIDIA Tesla P100 PCIe 16GB ；系统 ubuntu22.04| NVIDIA Tesla P100 PCIe 16GB; 系统ubuntu22.04  | NVIDIA Tesla P100 PCIe 16GB; 系统ubuntu22.04  |
| 上传日期        | 2025-8-31                                     | 2025-8-31                                     | 2025-8-31                                    |
| MindSpore版本   | 1.8.1                                         | 1.8.1                                         | 1.8.1                                        |
| MindQuantum版本 | 0.7.0                                         | 0.7.0                                         | 0.7.0                                        |
| 数据集          | warship                                       | mnist                                         | thucnews                                     |
| 训练参数        | epoch=30, steps per epoch=41, batch_size = 10 | epoch=10.steps per epoch=116, batch_size = 32 | epoch=7.steps per epoch=200, batch_size = 16|
| 优化器          | Adam                                          | Adam                                          | Adam                                         |
| 损失函数        | Binary CrossEntropy Loss                      | Binary CrossEntropy Loss                      | CrossEntropy Loss                            |
| 输出            | accuracy                                      | accuracy                                      | accuracy                                     |
| 精度            | 82.00±1.05%                                   | 98.98±0.22%                                   | 96.13%                                       |
| 训练时长        | 1990.4min                                     | 998.8min                                      |2053min                                       |

# 随机情况说明

- 脚本dataset.py中，在创建舰船数据加载器时，对舰船数据进行打乱处理时，设置了随机数种子

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
 # 其他情况说明

- 为了保证论文的隐私，本项目只上传了部分代码，缺失的部分包括eswo.py(用于实现缓存机制和增强蛛蜂优化算法来寻找优秀的QNN),dataset.py(三个数据集的预处理过程)，
  weak_train_data文件夹(包括预训练的QNN之间的编码对以及二元优劣关系)以及common.py(含噪量子电路的构建过程)，当论文发表后，我们会补充上述关键文件，请读者理解。
