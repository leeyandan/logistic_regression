# Logistic Regression 逻辑回归
> 参考李航《统计学习方法》第六章
## 1. Logistic distribution 
逻辑斯蒂分布，主要就是逻辑斯蒂分布函数F。因为其函数值在[0, 1]之间，符合概率的分布特点，良好的对称性也是选择它的一个原因。<br>
一般的逻辑斯蒂分布函数如下：

$$ F(x)= \frac{1}{1+e^{-(x-\mu)/\gamma }} \tag{1.1} $$
其实 $\mu$ 为位置参数， F(x) 关于点 $(\mu, \frac{1}{2})$ 中心对称。 $\gamma >0$ 为形状参数，值越小，曲线在中心附近增长越快。<br>
当 $\mu=0 \ \gamma=1$ 时，分布函数F为

$$ F(x)= \frac{1}{1+e^{-x}} \tag{1.2} $$
这个就是我们常用的sigmoid函数 （S形曲线）。

## 2. 二项Logistic 回归模型
对于二分类的模型，有如下的条件概率分布：

$$p(y=1|x) = \frac{1}{1+e^{-wx}} \tag{1.3}$$
<br>

$$p(y=0|x) =1-p(y=1|x) =\frac{1}{1+e^{wx}} \tag{1.4}$$
其实 $w\ x$ 都是拓展后的向量。$w=(w^{(1)},w^{(2)},w^{(3)},...,w^{(n)},b)\ x=(x^{(1)},x^{(2)},x^{(3)},...,x^{(n)},1)$

在这个模型中，它只是做了两件事儿：<br>
1. 线性变换：$wx+b$
2. 将线性变换的值代入sigmoid函数中
## 3. 二项模型参数估计
### 3.1 参数估计的过程就是**最大似然估计**+**最优化问题**<br>
训练数据 $T=\{(x_1, y_1),\ (x_2, y_2),\ (x_3, y_3),\ ...,\ (x_n, y_n)\ \}$<br>
似然函数为

$$\prod_{i=1}^{n}P(y=1|x_i)^{y_i}\ {(\ 1-P(y=1|x_i)\ )}^{1-y_i} \tag{1.5}$$
对数似然函数为

$$ 
\begin{aligned}L(w) =& \sum_{i=1}^{n}[y_i\ logP(y=1|x_i)+(1-y_i)\ log(1-log\ P(y=1|x_i)))] \\
=& \sum_{i=1}^{n}[y_i\ (w*x_i)- log(1+e^{w*x_i})] \tag{1.6}
\end{aligned}
$$

### 3.2 梯度下降法最大化公式（1.6）
>书中并没有给出具体的优化过程，我要来试试试能不能掌握我的梯度下降法啦啦啦啦！！
1. 梯度下降法<br>
对于优化问题:

$$min\ f(x)$$
&emsp;&emsp;&emsp;
梯度$\nabla\ f(x)$就是函数f在x处增长最快的方向，反之，负梯度$-\nabla\ f(x)$就是减少最快的方向。<br>
&emsp;&emsp;&emsp;所以在这个优化问题中可以用迭代至收敛来优化它。

$$x^{i+1} = x^{i}-\lambda*\nabla\ f(x^i)$$

2. 梯度下降优化logistic&ensp;回归<br>
优化目标（公式1.6）：

$$L(w)= \sum_{i=1}^{n}\left [\ y_i\ (w*x_i)- log(1+e^{w*x_i})\ \right ]$$
&emsp;&emsp;&emsp; 其中 $y_i=\{\ 0,1\ \}\ \ x_i=\{\ x_i^1,x_i^2,x_i^3,\dots,x_i^D \ \}\ w=\{w_1,w_2,\dots\ ,w_D\}$ <br>
&emsp;&emsp;&emsp; 令 $t_i = \sum_{d=1}^{D}w_d*x_i^d$ 得：

$$
\begin{aligned}
L&= \sum_{i=1}^{n}\left [\ y_i\ t_i- log(1+e^{t_i})\ \right ] \\
t_i& = \sum_{d=1}^{D}w_d*x_i^d
\end{aligned}
$$
&emsp;&emsp;&emsp; 所以求导得：

$$
\begin{aligned}
\frac{\partial L}{\partial w_d} = \frac{\partial L}{\partial t_i}\frac{\partial t_i}{\partial w_d}=& \left ( y_i-\frac{e^{t_i}}{1+e^{t_i}} \right )*x_i^d\\
=& \left ( y_i-\frac{e^{\sum_{d=1}^{D}w_d*x_i^d}}{1+e^{\sum_{d=1}^{D}w_d*x_i^d}} \right )*x_i^d
\end{aligned} \tag{1.7}
$$

&emsp;&emsp;&emsp; 综上，$w_d$ 的更新如下：

$$
\begin{aligned}
w_d^{k+1} =&\ w_d^{k} + \lambda \frac{\partial L}{\partial w_d}\\
=&\ w_d^{k} + \lambda*\left ( y_i-\frac{e^{\sum_{d=1}^{D}w_d*x_i^d}}{1+e^{\sum_{d=1}^{D}w_d*x_i^d}} \right )*x_i^d
\end{aligned} \tag{1.8}
$$
&emsp;&emsp;&emsp; 其中，$\lambda$ 为学习率。








### 3.3 拟牛顿法
> 再学点优化算法呗。。。



## 4. 多项Logistic 回归模型

## 5. 多项模型回归估计
