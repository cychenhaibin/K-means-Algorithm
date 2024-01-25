## 注：此代码已在CQUPT数据工程查重代码库中，仅供参考，请勿照抄
### 一、实验题目及内容
完成K-means算法的代码实现（同时提交源代码）及数据（至少测试5个数据集，数据集来源建议采用UCI数据集）测试结果。
### 二、实验过程步骤（注意是主要关键步骤，适当文字+截图说明）、实验结果及分析
#### 实验完整代码：
<img width="442" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/eb93fd02-e0c9-49b9-a562-66d3b9b5f412">
<img width="422" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/397a0536-e932-4655-9680-426f1e42f6f2">
<img width="442" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/98e9c047-a55c-4ec5-bf4b-6a10b3e3ccf5">

#### 1.k-means算法介绍
k均值聚类算法（k-means clustering algorithm）是一种迭代求解的聚类分析算法，其步骤是，预将数据分为K组，则随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。聚类中心以及分配给它们的对象就代表一个聚类。每分配一个样本，聚类的聚类中心会根据聚类中现有的对象被重新计算。这个过程将不断重复直到满足某个终止条件。终止条件可以是没有（或最小数目）对象被重新分配给不同的聚类，没有（或最小数目）聚类中心再发生变化，误差平方和局部最小。

#### 2.算法步骤
①对于给定的一组数据，随机初始化K个聚类中心（簇中心）
②计算每个数据到簇中心的距离（一般采用欧氏距离），并把该数据归为离它最近的簇。
③根据得到的簇，重新计算簇中心。
④对步骤2、步骤3进行迭代直至簇中心不再改变或者小于指定阈值。

#### 3.K-means算法流程
              <img width="237" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/6beb6911-34da-4695-917f-73ed7efe6674">
   
#### 4.K-means伪代码
<img width="448" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/b6f92451-bdd8-4c98-9ec3-5d561092d49c">
曼哈顿距离公式：
         <img width="226" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/ff5fa1a1-cdeb-4a4c-ab4f-b3e88f473b27">  
欧几里得距离公式：
                <img width="137" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/8c30fb09-f70b-45bf-b09a-b37854fc3b7f">
     
#### 5.k-means核心代码解析
①关于k-means在Windows上使用MKL（Math Kernel Library）时可能会导致内存泄漏的警告。该警告建议通过设置环境变量OMP_NUM_THREADS=1来避免这个问题。
    在Python中，在开头添加以下代码来设置环境变量：
   <img width="413" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/f4b988cd-5cdf-4e09-b2d2-f0dae83802f9">
    确保在使用k-means时只使用一个线程，从而避免可能的内存泄漏问题。
    
##### ②欧氏距离函数(distance):
该函数计算两个向量之间的欧氏距离，用于度量样本点之间的相似性。
<img width="409" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/04d10d4c-e309-42fa-bee8-107e94983cce">
  
##### ③K均值聚类算法 (kmeans):   
通过迭代更新质心，将样本点分配到最近的质心，并最终形成K个聚类。
使用"手肘法"确定最佳的K值，即聚类数。
<img width="409" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/a7398935-8e48-4487-a780-64953a3f8d13">

##### ④手肘法 (elbow_method):
通过尝试不同的K值，计算每个K值下聚类的畸变程度（Distortion）。
绘制K值与畸变程度的图表，以帮助选择最优的K值。
<img width="408" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/a47b44a6-34eb-4da2-865f-a48ebcdf15a4">
<img width="410" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/6b77efe7-9339-428b-bc28-70e631c001fd">

##### ⑤数据集加载和处理:
使用datasets模块加载鸢尾花、葡萄酒、数字、乳腺癌和糖尿病数据集。
通过PCA降维，将数据集的特征维度减少到2，以便在二维空间中可视化。
<img width="409" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/581a4b25-644d-4531-9293-29ad3dd4f2ab">

##### ⑥循环处理多个数据集:   
对每个数据集，使用手肘法找到最佳K值，然后应用K均值聚类算法。
最后，通过散点图可视化聚类结果，标记质心。
<img width="409" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/c39ef88b-c918-481c-9906-94879b4509a7">
<img width="408" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/a3b9d065-d1b5-409e-9c73-046f1123ce92">

#### 6.实验结果
<img width="206" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/d7fe1d4c-262b-4f80-898b-6fedf7d9ac97">
鸢尾花数据集K值折线图
<img width="227" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/2c44369b-fd5a-48e0-8c3e-b95051249867">
鸢尾花数据集聚类效果图 
<img width="209" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/37206170-8027-4ba4-a998-4e1a391a3d7e">

<img width="188" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/b9923475-96c7-41d2-ac3e-8a28a4436a50">
葡萄酒数据集K值折线图   
<img width="188" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/28b3a14a-ef57-464f-9cf1-96dc9a93524d">
葡萄酒数据集聚类效果图 
<img width="206" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/43f7994f-f31f-45c4-8180-195ae25b757e">

<img width="211" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/02ea518e-5d77-4d27-b1b9-de7a9b5d2d87">
手写数字数据集K值折线图  
<img width="229" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/904727fb-eb13-447b-9a17-89cc41c1e4eb">
手写数字数据集聚类效果图  
<img width="216" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/32cb981b-f008-462e-8785-fcfe555a6f19">
  
<img width="287" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/60a831c4-5c4f-4766-84b8-423ec50bdc27">
乳腺癌数据集K值折线图 
<img width="225" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/d365453d-ee0e-4ece-a900-5e2d7760c55c">
乳腺癌数据集聚类效果图    
<img width="220" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/c8253628-76e3-42f5-a7f0-fefb8f7d9f6f">

<img width="244" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/82ff3fbb-c5c1-4fce-a475-841d07f3c3c2">
糖尿病数据集K值折线图 
<img width="224" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/80ceb9a3-6f88-4c98-89b2-fb75c3949943">
糖尿病数据集聚类效果图    
<img width="220" alt="image" src="https://github.com/cychenhaibin/K-means-Algorithm/assets/117504781/4793dd28-79b8-43d4-aa8c-5490b227fcf7">

#### 7.实验结果分析
k-means算法的优缺点
优点：
1、简单易懂：代码结构清晰，易于理解和修改。
2、功能完整： 实现了K均值聚类算法、手肘法确定最佳K值以及数据集的加载、处理和可视化。
3、灵活性： 可以轻松处理多个不同特征维度的数据集，并通过PCA进行降维以便可视化。
缺点：
1、手肘法选取K值：代码中使用了手肘法来选择最佳的K值，但手肘法有时并不总是最佳的选择，有些情况下可能需要其他评估指标来确定最佳的聚类数。
2、聚类效果评估： 代码中缺少对聚类效果的定量评估，如轮廓系数等，这些评估指标可以帮助更好地评估聚类结果的质量。
3、内存泄漏警告： 代码中可能会出现与MKL和内存泄漏相关的警告，这可能会影响程序的性能和稳定性。
