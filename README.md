# Logistic Regression 逻辑回归
> 参考李航《统计学习方法》第六章
## 1. 源代码
+ code\binary_logistic_gression.py
    + 说明：二分类逻辑回归代码实现
    + 数据集：data\Magic Dataset.txt
    + 十次测试平均准确度：0.790

+ code\multi_logistic_gression.py
    + 说明：多分类逻辑回归代码实现
    + 数据集：data\ecoli-3c.txt
    + 十次测试平均准确度：0.882
## 2. 文档说明
+ [可以参考博客，理论推导过程都在上面写得很详细啦](https://blog.csdn.net/yan_dan/article/details/103335527)
## 3. 数据集
> 数据集格式N+1类型就可以直接用，模型要求的类标用123数字化表示，数据格式由prepare_data_x_y()函数控制。
+ data\ecoli-3c.txt
    + 大肠杆菌数据集，[来源以及详细说明](http://archive.ics.uci.edu/ml/datasets/Ecoli)
    + 属性数量：8
    + 类别数量：5
    + 作用：多分类任务测试数据集
+ data\Magic Dataset.txt
    + Gamma粒子数据集，[来源以及详细说明](http://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)
    + 属性数量：10
    + 类别数量：2
    + 作用：二分类任务测试数据集