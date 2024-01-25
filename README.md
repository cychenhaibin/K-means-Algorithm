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
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/73f3bf14-54ed-4153-a8dc-2d928acf318d)

#### 4. K-means伪代码
```
输入 n 个数据对象集合Xi ;输出 k 个聚类中心 Zj 及K 个聚类数据对象集合 Cj .
Procedure K -means(s , k)
S ={x 1 ,x 2 , …,x n };
m =1;for j =1 to k 初始化聚类中心 Zj ;
do {for i =1 to n
　　for j=1 to k
　　　{D(Xi ,Zj)= Xi -Zj ;if D(Xi ,Zj)=Min{D(Xi ,Zj)}then Xi ∈Cj ;}//归类
　　　if m=1 then Jc(m)=∑kj=1∑ Xi -Zj
2
　　m =m+1;for j =1 to k
　　Zj =(∑
n
i=1 (Xi)
j )/n;//重置聚类中心
　　}while J c (m)-J c (m -1) >ξ
```
曼哈顿距离公式：
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/7bac8ffa-de19-4f6a-80a3-5e4733f0485a)
欧几里得距离公式：
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/d736cfa1-c300-4bed-87ce-2f7ff9a1c784)

#### 5. k-means核心代码解析

①关于k-means在Windows上使用MKL（Math Kernel Library）时可能会导致内存泄漏的警告。该警告建议通过设置环境变量OMP_NUM_THREADS=1来避免这个问题。
    在Python中，在开头添加以下代码来设置环境变量：
 ```
# 设置环境变量 OMP_NUM_THREADS=1
# 当块数少于可用线程时，KMeans 在使用 MKL 的Windows上会出现内存泄漏。
# 通过设置环境变量 OMP_NUM_THREADS=1 来避免它。
# 导入 kmeans 之前
import os
os.environ["OMP_NUM_THREADS"] = '1'
 ```
确保在使用k-means时只使用一个线程，从而避免可能的内存泄漏问题。

##### ②欧氏距离函数(distance):

该函数计算两个向量之间的欧氏距离，用于度量样本点之间的相似性。
```
# def distance(x, y):
#     # 计算两向量间的欧氏距离
#     return np.linalg.norm(x - y)

# 计算两向量间的欧氏距离
def distance(X1, X2):
    result = 0
    for (x1, x2) in zip(X1, X2):
        result += (x1 - x2) ** 2
    return np.sqrt(result)
```

##### ③K均值聚类算法 (kmeans):   

通过迭代更新质心，将样本点分配到最近的质心，并最终形成K个聚类。
使用"手肘法"确定最佳的K值，即聚类数。
```
def kmeans(data, k):
    # k均值聚类算法
    kmeans_model = KMeans(n_clusters=k, n_init='auto')
    kmeans_model.fit(data)
    return (kmeans_model.labels_,
            kmeans_model.cluster_centers_)
```

##### ④手肘法 (elbow_method):

通过尝试不同的K值，计算每个K值下聚类的畸变程度（Distortion）。
绘制K值与畸变程度的图表，以帮助选择最优的K值。
```
def elbow_method(data):
    # 手肘法确定最佳 k 值
    distortions = []
    K_range = range(1, 11)

    for k in K_range:
        labels, centers = kmeans(data, k)
        distortion = sum(np.min(
            distance(
                data[i],
                centers[labels[i]]
            )**2) for i in range(len(data)))
        distortions.append(distortion)

    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()
```

##### ⑤数据集加载和处理:

使用datasets模块加载鸢尾花、葡萄酒、数字、乳腺癌和糖尿病数据集。
通过PCA降维，将数据集的特征维度减少到2，以便在二维空间中可视化。
```
def load_and_process_dataset(dataset):
    # 加载处理数据集
    data = dataset.data
    labels = dataset.target

    # 使用 PCA 降维可视化
    reduced_data = PCA(n_components=2).fit_transform(data)

    return reduced_data, labels
```

##### ⑥循环处理多个数据集:   

对每个数据集，使用手肘法找到最佳K值，然后应用K均值聚类算法。
最后，通过散点图可视化聚类结果，标记质心。
```
def visualize_clusters(data, labels, centers, title):
    # 可视化聚类结果
    plt.scatter(data[:, 0],
                data[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.5)
    plt.scatter(centers[:, 0],
                centers[:, 1],
                c='red',
                marker='X',
                s=200,
                label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

# 加载数据集
iris = datasets.load_iris()
wine = datasets.load_wine()
digits = datasets.load_digits()
breast_cancer = datasets.load_breast_cancer()
diabetes = datasets.load_diabetes()

datasets_list = [(iris, "Iris"),
                 (wine, "Wine"),
                 (digits, "Digits"),
                 (breast_cancer, "Breast Cancer"),
                 (diabetes, "Diabetes")]

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
    visualize_clusters(data,
                       cluster_labels,
                       cluster_centers,
                       f'K-means Clustering - {name}')
```

#### 6.实验结果
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/4f6f10bf-bfbf-436a-9dbc-e86ecf4423b8)<br>
鸢尾花数据集K值折线图<br>
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/b67f9624-49b5-473a-8eea-b74fe0f34c8b)<br>
鸢尾花数据集聚类效果图<br> 
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/e6314ba7-d02d-4abf-b077-3ed204840ceb)<br>

![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/b25108a9-0996-4de2-9606-29f27aeddc97)<br>
葡萄酒数据集K值折线图<br>  
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/a333a10c-1c56-456c-b979-25605a949aa1)<br>
葡萄酒数据集聚类效果图<br> 
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/ede6ab92-83f8-4917-a61c-bb3fc33b11bc)<br>

![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/dd5ecb2b-4276-4479-9739-bc3ba810d9ae)<br>
手写数字数据集K值折线图<br>
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/ca8f8711-2147-4695-a541-1a27de811be6)<br>
手写数字数据集聚类效果图<br>
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/aa754a2e-0919-4ce4-bcf1-5132cf44692f)<br>

![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/5eb719a9-e96d-432a-ba19-f09c3892b057)<br>
乳腺癌数据集K值折线图<br> 
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/07cbbcbd-4a83-4a9a-a77b-39fb3358ea5d)<br>
乳腺癌数据集聚类效果图<br>
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/010341e0-a35c-4eba-9489-949e65db68de)<br>

![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/f804f28a-ce84-43f6-9f12-f7d05e7df8b7)<br>
糖尿病数据集K值折线图<br> 
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/207f84f2-eaaa-4bfc-ac28-cc6d998cd8b2)<br>
糖尿病数据集聚类效果图<br>
![image](https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/3bd35ae7-e14e-4998-975e-8bc86b139890)<br>

#### 7.实验结果分析

k-means算法的优缺点<br>
优点：<br>
1、简单易懂：代码结构清晰，易于理解和修改。<br>
2、功能完整： 实现了K均值聚类算法、手肘法确定最佳K值以及数据集的加载、处理和可视化。<br>
3、灵活性： 可以轻松处理多个不同特征维度的数据集，并通过PCA进行降维以便可视化。<br>
缺点：<br>
1、手肘法选取K值：代码中使用了手肘法来选择最佳的K值，但手肘法有时并不总是最佳的选择，有些情况下可能需要其他评估指标来确定最佳的聚类数。<br>
2、聚类效果评估： 代码中缺少对聚类效果的定量评估，如轮廓系数等，这些评估指标可以帮助更好地评估聚类结果的质量。<br>
3、内存泄漏警告： 代码中可能会出现与MKL和内存泄漏相关的警告，这可能会影响程序的性能和稳定性。
