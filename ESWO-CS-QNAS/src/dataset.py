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
"""Dataset loaders."""
import os
import glob
import time
import numpy as np
import mindspore as ms
from mindspore.dataset import vision, NumpySlicesDataset
from mindspore.dataset.vision import Inter
from mindvision.dataset import Mnist
from src.utils.data_preprocess import filter_36, inver_label, remove_contradicting, binary_image, flatten_image
from gensim.models import Word2Vec
from PIL import Image
import jieba
import random
import re


def create_loaders(cfg):
    """Dataset loader"""
    if cfg.DATASET.type == "mnist":
        train_dataset, eval_dataset = create_mnist_loaders(cfg)
    elif cfg.DATASET.type == "warship":
        train_dataset, eval_dataset = create_warship_loaders(cfg)
    elif cfg.DATASET.type == "thucnews":
        train_dataset, eval_dataset = create_thucnews_loaders(cfg)
    else:
        raise ValueError("cfg.DATASET.type must be mnist, warship or thucnews")
    return train_dataset, eval_dataset


def create_mnist_loaders(config):
    # Load mnist data set (training set, eval set)
    train_data = Mnist(
        path=config.DATASET.path,
        split='train',
        shuffle=False,
        download=True
    )
    test_data = Mnist(
        path=config.DATASET.path,
        split='test',
        shuffle=False,
        download=True
    )
    train_dataset = train_data.dataset
    eval_dataset = test_data.dataset

    # Scale and normalize the image pixels
    train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
    eval_dataset = eval_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')

    # Filter numbers 3 and 6
    train_dataset = train_dataset.filter(predicate=filter_36, input_columns=['label'])
    eval_dataset = eval_dataset.filter(predicate=filter_36, input_columns=['label'])

    # Convert (3,6) labels to (0,1)
    train_dataset = train_dataset.map(operations=[inver_label], input_columns=['label'])
    eval_dataset = eval_dataset.map(operations=[inver_label], input_columns=['label'])

    # Bilinear interpolation shrinks the image size to 4 * 4
    train_dataset = train_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='image')
    eval_dataset = eval_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='image')

    # Remove contradictory data caused by shrinking pictures
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)

    # Image binarization processing
    train_dataset = train_dataset.map(operations=[binary_image], input_columns='image')
    eval_dataset = eval_dataset.map(operations=[binary_image], input_columns='image')

    # Remove contradictory data caused by binarization
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)

    # Save data preprocessed results
    if not os.path.exists(config.DATASET.path + 'pretreatment/'):
        os.system("mkdir -p " + config.DATASET.path + 'pretreatment/')
        train_dataset.save(config.DATASET.path + 'pretreatment/train_dataset')
        eval_dataset.save(config.DATASET.path + 'pretreatment/eval_dataset')
    train_dataset, eval_dataset = 0, 1
    return train_dataset, eval_dataset


def create_warship_loaders(config):
    # Load warship data set (training set, eval set)
    images = []
    for f in glob.iglob(config.DATASET.path + "Burke/*"):
        images.append(np.asarray(Image.open(f).convert('L')))
    for f in glob.iglob(config.DATASET.path + "Nimitz/*"):
        images.append(np.asarray(Image.open(f).convert('L')))
    images_test = []
    tmp=0
    for f in glob.iglob(config.DATASET.path + "test_burke/*"):
        images_test.append(np.asarray(Image.open(f).convert('L')))
        tmp=tmp+1
        if tmp==78:
            break
    for f in glob.iglob(config.DATASET.path + "test_nimitz/*"):
        images_test.append(np.asarray(Image.open(f).convert('L')))
        tmp=tmp+1
        if tmp==150:
            break
    # Scale and normalize the image pixels
    images = np.array(images)
    images = images / 255
    images_test = np.array(images_test)
    images_test = images_test / 255
    # Generate label, Burke is 1, Nimitz is 0
    train_label = np.array([])
    test_label = np.array([])
    for i in range(202):
        i += 1
        i -= 1
        train_label = np.append(train_label, 1)
    for i in range(209):
        i += 1
        i -= 1
        train_label = np.append(train_label, 0)
    for i in range(78):
        i += 1
        i -= 1
        test_label = np.append(test_label, 1)
    for i in range(72):
        i += 1
        i -= 1
        test_label = np.append(test_label, 0)
    # Generate dataset from images and tags
    train_dataset = NumpySlicesDataset({'features': images, 'labels': train_label}, shuffle=False)
    eval_dataset = NumpySlicesDataset({'features': images_test, 'labels': test_label}, shuffle=False)
    # Bilinear interpolation reduces the image size to 4 * 4
    train_dataset = train_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='features')
    eval_dataset = eval_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='features')
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)
    train_dataset = train_dataset.map(operations=[flatten_image], input_columns='features')
    eval_dataset = eval_dataset.map(operations=[flatten_image], input_columns='features')
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)
    # shuffle dataset
    ms.dataset.config.set_seed(1234)
    train_dataset = train_dataset.shuffle(train_dataset.get_dataset_size())
    eval_dataset = eval_dataset.shuffle(eval_dataset.get_dataset_size())
    ms.dataset.config.set_seed(int(time.time()))
    # Save data preprocessed results
    if not os.path.exists(config.DATASET.path + 'pretreatment/'):
        os.system("mkdir -p " + config.DATASET.path + 'pretreatment/')
        train_dataset.save(config.DATASET.path + 'pretreatment/train_dataset')
        eval_dataset.save(config.DATASET.path + 'pretreatment/eval_dataset')
    train_dataset, eval_dataset = 0, 1
    return train_dataset, eval_dataset
def create_thucnews_loaders(config):
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    # four news categories
    TARGET_CATEGORIES = ["体育", "财经", "娱乐", "科技"]
    SAMPLES_PER_CATEGORY = 1000
    WORD2VEC_DIM = 16
    TEST_RATIO = 0.2

    def load_stopwords(stopwords_path=config.DATASET.path+"stopwords.txt"):
        """加载停用词表"""
        if not os.path.exists(stopwords_path):
            print(f"未找到停用词表{stopwords_path}，将不使用停用词过滤")
            return set()
        
        with open(stopwords_path, "r", encoding="utf-8") as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)

    def preprocess_text(text, stopwords):
        """文本预处理：去除特殊字符、分词、过滤停用词"""
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 分词
        words = jieba.cut(text.strip())
        # 过滤停用词和空词
        words = [word for word in words if word.strip() and word not in stopwords]
        return words
    def int_to_binary_label(label, num_bits=2):
        """interger label to binary label"""
        binary_str = bin(label)[2:]
        binary_str = binary_str.zfill(num_bits)
        return [int(bit) for bit in binary_str]
    def load_data(file_path, target_categories, max_samples_per_category):
        """load data and extract target categories"""
        category_data = {}
        # Initialize the list for each target category
        for category in target_categories:
            category_data[category] = []
        stopwords = load_stopwords()
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                    
                category, content = parts[0], parts[1]
                
                # 只处理目标类别
                if category in target_categories:
                    # 文本预处理
                    words = preprocess_text(content, stopwords)
                    if words:  # 确保预处理后不为空
                        category_data[category].append(words)
                        
                        # 达到指定数量则停止该类别收集
                        if len(category_data[category]) >= max_samples_per_category:
                            # 检查是否所有目标类别都已收集足够样本
                            all_collected = True
                            for cat in target_categories:
                                if len(category_data[cat]) < max_samples_per_category:
                                    all_collected = False
                                    break
                            if all_collected:
                                break
        
        # 截断到指定数量，并转换为标签和文本的列表
        texts = []
        labels = []
        num_bits = len(bin(len(target_categories) - 1)) - 2  # 减去2是因为bin()返回的字符串包含'0b'前缀
        print(f"使用{num_bits}位二进制表示标签")
        for idx, category in enumerate(target_categories):
            samples = category_data[category][:max_samples_per_category]
            texts.extend(samples)
            binary_label = int_to_binary_label(idx, num_bits)
            labels.extend([binary_label] * len(samples))  # 为每个类别分配二值标签
            
            print(f"加载类别 {category}：{len(samples)} 条数据，二值标签为 {binary_label}")
            print(f"加载类别 {category}：{len(samples)} 条数据")
        
        return texts, labels, target_categories

    def train_word2vec(texts, vector_size=16, window=5, min_count=1, epochs=10):
        """训练Word2Vec模型"""
        print(f"开始训练Word2Vec模型，向量维度：{vector_size}")
        model = Word2Vec(
            sentences=texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )
        print("Word2Vec模型训练完成")
        return model

    def text_to_vector(text, model, vector_size):
        """将文本转换为向量（词向量的平均值）"""
        vectors = [model.wv[word] for word in text if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def split_train_test(texts, labels, test_ratio=0.2, random_seed=42):
        """划分训练集和测试集"""
        # 打乱数据
        combined = list(zip(texts, labels))
        random.seed(random_seed)
        random.shuffle(combined)
        texts_shuffled, labels_shuffled = zip(*combined)
        
        # 计算分割点
        split_idx = int(len(texts_shuffled) * (1 - test_ratio))
        
        # 分割数据
        train_texts, test_texts = texts_shuffled[:split_idx], texts_shuffled[split_idx:]
        train_labels, test_labels = labels_shuffled[:split_idx], labels_shuffled[split_idx:]
        
        print(f"数据集划分完成：训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条")
        return train_texts, train_labels, test_texts, test_labels

    if not os.path.exists(config.DATASET.path + 'pretreatment/'):
        # 数据文件路径（请根据实际路径修改）
        data_dir = config.DATASET.path  # THUCNews数据集所在目录
        train_file = os.path.join(data_dir, "cnews_train.txt")
        
        # 1. 加载并预处理数据
        print("开始加载数据...")
        texts, labels, categories = load_data(
            train_file, 
            TARGET_CATEGORIES, 
            SAMPLES_PER_CATEGORY
        )
        
        # 2. 训练Word2Vec模型
        word2vec_model = train_word2vec(texts, vector_size=WORD2VEC_DIM)
        
        # 3. 将文本转换为向量
        print("正在将文本转换为向量...")
        text_vectors = [text_to_vector(text, word2vec_model, WORD2VEC_DIM) for text in texts]
        
        # 4. 划分训练集和测试集
        X_train, y_train, X_test, y_test = split_train_test(
            text_vectors, 
            labels, 
            test_ratio=TEST_RATIO
        )
        print(type(X_train))
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_test=np.array(X_test)
        y_test=np.array(y_test)
        print(type(X_train))
        # 构建训练数据集迭代器
        train_dataset = ms.dataset.NumpySlicesDataset(
            (X_train, y_train),
            column_names=["x", "y"],
            shuffle=True,
        )
        eval_dataset = ms.dataset.NumpySlicesDataset(
            (X_test, y_test),
            column_names=["x", "y"],
            shuffle=True,
        )
    # Save data preprocessed results
        os.system("mkdir -p " + config.DATASET.path + '/pretreatment/')
        train_dataset.save(config.DATASET.path + 'pretreatment/train_dataset')
        eval_dataset.save(config.DATASET.path + 'pretreatment/eval_dataset')
    train_dataset, eval_dataset = 0, 1
    return train_dataset, eval_dataset