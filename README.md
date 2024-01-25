## 注：此代码已在CQUPT数据工程查重代码库中，仅供参考，请勿照抄

### 一、实验题目及内容

完成K-means算法的代码实现（同时提交源代码）及数据（至少测试5个数据集，数据集来源建议采用UCI数据集）测试结果。

### 二、实验过程步骤（注意是主要关键步骤，适当文字+截图说明）、实验结果及分析

#### 实验完整代码：
```
# 设置环境变量 OMP_NUM_THREADS=1
# 当块数少于可用线程时，KMeans 在使用 MKL 的Windows上会出现内存泄漏。
# 通过设置环境变量 OMP_NUM_THREADS=1 来避免它。
# 导入 kmeans 之前
import os
os.environ["OMP_NUM_THREADS"] = '1'

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# def distance(x, y):
#     # 计算两向量间的欧氏距离
#     return np.linalg.norm(x - y)

# 计算两向量间的欧氏距离
def distance(X1, X2):
    result = 0
    for (x1, x2) in zip(X1, X2):
        result += (x1 - x2) ** 2
    return np.sqrt(result)

def kmeans(data, k):

    # k均值聚类算法
    kmeans_model = KMeans(n_clusters=k, n_init='auto')
    kmeans_model.fit(data)
    return (kmeans_model.labels_,
            kmeans_model.cluster_centers_)

def elbow_method(data):
    # 手肘法确定最佳 k 值
    distortions = []
    K_range = range(1, 11)

    for k in K_range:
        labels, centers = kmeans(data, k)
        distortion = sum(np.min(distance(data[i],centers[labels[i]])**2) for i in range(len(data)))
        distortions.append(distortion)

    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()

def load_and_process_dataset(dataset):
    # 加载处理数据集
    data = dataset.data
    labels = dataset.target

    # 使用 PCA 降维可视化
    reduced_data = PCA(n_components=2).fit_transform(data)

    return reduced_data, labels

def visualize_clusters(data, labels, centers, title):
    # 可视化聚类结果
    plt.scatter(data[:, 0],data[:, 1],c=labels,cmap='viridis',alpha=0.5)
    plt.scatter(centers[:, 0],centers[:, 1],c='red',marker='X',s=200,label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

# 加载数据集
iris = datasets.load_iris()
wine = datasets.load_wine()
digits = datasets.load_digits()
breast_cancer = datasets.load_breast_cancer()
diabetes = datasets.load_diabetes()

datasets_list = [(iris, "Iris"),(wine, "Wine"),(digits, "Digits"),(breast_cancer, "Breast Cancer"),(diabetes, "Diabetes")]

# 处理和分析每个数据集
for dataset, name in datasets_list:
    data, labels = load_and_process_dataset(dataset)

    # 手肘法确定最佳 k 值
    elbow_method(data)

    # 手动选择手肘法得到的最佳 k 值
    optimal_k = int(input(f"输入 {name} 的最佳K值: "))

    # 应用 k 均值聚类
    cluster_labels, cluster_centers = kmeans(data, optimal_k)

    # 可视化聚类结果
    visualize_clusters(data,cluster_labels,cluster_centers,f'K-means Clustering - {name}')
```

#### 1. k-means算法介绍

k均值聚类算法（k-means clustering algorithm）是一种迭代求解的聚类分析算法，其步骤是，预将数据分为K组，则随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。聚类中心以及分配给它们的对象就代表一个聚类。每分配一个样本，聚类的聚类中心会根据聚类中现有的对象被重新计算。这个过程将不断重复直到满足某个终止条件。终止条件可以是没有（或最小数目）对象被重新分配给不同的聚类，没有（或最小数目）聚类中心再发生变化，误差平方和局部最小。

#### 2. 算法步骤

①对于给定的一组数据，随机初始化K个聚类中心（簇中心）
②计算每个数据到簇中心的距离（一般采用欧氏距离），并把该数据归为离它最近的簇。
③根据得到的簇，重新计算簇中心。
④对步骤2、步骤3进行迭代直至簇中心不再改变或者小于指定阈值。

#### 3. K-means算法流程
![image-20240125193918742](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240125193918742.png)
