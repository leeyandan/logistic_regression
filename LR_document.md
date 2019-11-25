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

$$p(y=1|x) = \frac{1}{1+e^{-wx}} \tag{2.1}$$
<br>

$$p(y=0|x) =1-p(y=1|x) =\frac{1}{1+e^{wx}} \tag{2.2}$$
其实 $w\ x$ 都是拓展后的向量。$w=(w^{(1)},w^{(2)},w^{(3)},...,w^{(n)},b)\ x=(x^{(1)},x^{(2)},x^{(3)},...,x^{(n)},1)$

在这个模型中，它只是做了两件事儿：<br>
1. 线性变换：$wx+b$
2. 将线性变换的值代入sigmoid函数中
## 3. 二项模型参数估计
### 3.1 参数估计的过程就是**最大似然估计**+**最优化问题**<br>
训练数据 $T=\{(x_1, y_1),\ (x_2, y_2),\ (x_3, y_3),\ ...,\ (x_n, y_n)\ \}$<br>
似然函数为

$$\prod_{i=1}^{n}P(y=1|x_i)^{y_i}\ {(\ 1-P(y=1|x_i)\ )}^{1-y_i} \tag{2.3}$$
对数似然函数为

$$ 
\begin{aligned}L(w) =& \sum_{i=1}^{n}[y_i\ logP(y=1|x_i)+(1-y_i)\ log(1-log\ P(y=1|x_i)))] \\
=& \sum_{i=1}^{n}[y_i\ (w*x_i)- log(1+e^{w*x_i})] \tag{2.4}
\end{aligned}
$$

### 3.2 梯度下降法最大化公式（1.6）
>书中并没有给出具体的优化过程，我要来试试试能不能掌握我的梯度下降法啦啦啦啦！！
1. 梯度下降法<br>
对于优化问题:

$$min\ f(x) \tag{3.1}$$
&emsp;&emsp;&emsp;
梯度$\nabla\ f(x)$就是函数f在x处增长最快的方向，反之，负梯度$-\nabla\ f(x)$就是减少最快的方向。<br>
&emsp;&emsp;&emsp;所以在这个优化问题中可以用迭代至收敛来优化它。

$$x^{i+1} = x^{i}-\lambda*\nabla\ f(x^i) \tag{3.2}$$

2. 梯度下降优化logistic&ensp;回归<br>
优化目标（公式1.6）：

$$L(w)= \sum_{i=1}^{n}\left [\ y_i\ (w*x_i)- log(1+e^{w*x_i})\ \right ] \tag{3.3}$$
&emsp;&emsp;&emsp; 其中 $y_i=\{\ 0,1\ \}\ \ x_i=\{\ x_i^1,x_i^2,x_i^3,\dots,x_i^D \ \}\ w=\{w_1,w_2,\dots\ ,w_D\}$ <br>
&emsp;&emsp;&emsp; 令 $t_i = \sum_{d=1}^{D}w_d*x_i^d$ 得：

$$
\begin{aligned}
L&= \sum_{i=1}^{n}\left [\ y_i\ t_i- log(1+e^{t_i})\ \right ] \\
t_i& = \sum_{d=1}^{D}w_d*x_i^d
\end{aligned}
\tag{3.4}
$$
&emsp;&emsp;&emsp; 所以求导得：

$$
\begin{aligned}
\frac{\partial L}{\partial w_d} = \frac{\partial L}{\partial t_i}\frac{\partial t_i}{\partial w_d}=& \left[ \sum_{i=1}^n\left ( y_i-\frac{e^{t_i}}{1+e^{t_i}} \right )\right] *x_i^d\\
=&  \sum_{i=1}^n \left[ \left ( y_i-\frac{e^{\sum_{d=1}^{D}w_d*x_i^d}}{1+e^{\sum_{d=1}^{D}w_d*x_i^d}} \right )*x_i \right]
\end{aligned} \tag{3.5}
$$

&emsp;&emsp;&emsp; 综上，$w_d$ 的更新如下：

$$
\begin{aligned}
w_d^{k+1} =&\ w_d^{k} + \lambda \frac{\partial L}{\partial w_d}\\
=&\ w_d^{k} + \lambda*  \sum_{i=1}^n \left[ \left ( y_i-\frac{e^{\sum_{d=1}^{D}w_d*x_i^d}}{1+e^{\sum_{d=1}^{D}w_d*x_i^d}} \right )*x_i \right]
\end{aligned} \tag{3.6}
$$
&emsp;&emsp;&emsp; 其中，$\lambda$ 为学习率。








### 3.3 拟牛顿法
> 再学点优化算法呗。。。



## 4. 多项Logistic 回归模型
### 4.1 类概率计算
多分类问题，设有K个类，其中有一个类K（最后一个）为主类别（其实谁为主类别都可以）。参数一共有 K-1 组 W。

$$
\begin{aligned}
&P(Y_i=c\ |\ x_i)=\frac{e^{w_c*x_i}}{1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}} \qquad c=\{1,2,\cdots,K-1\} \tag{4.1} \\

&P(Y_i=K\ |\ x_i)=\frac{1}{1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}} \\

\end{aligned}
$$

### 4.2 为什么这么计算概率
> 参考 [博客](https://blog.csdn.net/huangjx36/article/details/78056375)

大概的思路就是用K-1个参数已经能表示出K个类的概率了。也就是说主类K可以用另外K-1个参数表示出来。
## 5. 多项模型回归估计 
>下面用最大似然估计+梯度下降法对模型参数进行估计
### 5.1 定义示性函数
定义示性函数是为了将类标实值化（01化）。当然也可以采用one-hot编码的方式

$$
f_c(y_i)=\begin{cases}
1 & y_i\ = c \\
0 & y_i\ \neq c

\end{cases}
\tag{5.1}
$$
### 5.2 似然函数
似然函数

$$
\begin{aligned}
L^0(w) =&  \prod_{i=1}^{N}\prod_{j=1}^{K}P(Y_i=j|x_i)^{f_j(y_i)}
\end{aligned}
\tag{5.2}
$$
对数似然函数

$$
\begin{aligned}
L^1(w) =&\  ln\ L^0(w) \\
=&\ \sum_{i=1}^{N}\sum_{j=1}^{K}{f_j(y_i)} ln\ P(Y_i=j|x_i) \\
=&\ \sum_{i=1}^{N}\left( \sum_{j=1}^{K-1} f_j(y_i) ln\frac{e^{w_j*x_i}}{1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}} + f_K(y_i)ln\frac{1}{1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}}\right) \\
=&\ \sum_{i=1}^{N}\left( \sum_{j=1}^{K-1} f_j(y_i)w_j*x_i- \sum_{j=1}^{K} f_j(y_i)ln \left( 1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}\right) \right)
\end{aligned}
\tag{5.3}
$$
### 5.3 求导
对于参数 $w_q\ q=\{1,2,\cdots,K-1\}$，求导得：

$$
\frac{\partial L^1(w)}{\partial w_q} = \sum_{i=1}^{N}\left ( f_q(y_i)x_i-\sum_{j=1}^{K} f_j(y_i)*\frac{1}{\left( 1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}\right)}* e^{w_qx_i}*x_i \right ) \tag{5.4}
$$

对于参数 $w_K$，求导得：

$$
\frac{\partial L^1(w)}{\partial w_K} = \sum_{i=1}^{N}\left (-\sum_{j=1}^{K} f_j(y_i)*\frac{1}{\left( 1+\sum_{p=1}^{K-1}{e^{w_p*x_i}}\right)}* e^{w_qx_i}*x_i \right )
\tag{5.5}
$$

### 5.4 迭代更新
对于参数 $w_q\ q=\{1,2,\cdots,K-1\}$，迭代式如下：

$$
w_q^{(k+1)} = w_q^{k} + \lambda*\frac{\partial L^1(w)}{\partial w_q^{k}}
\tag{5.6}
$$
>$\lambda$ 是学习率，$\frac{\partial L^1(w)}{\partial w_q^{k}}$ 由公式5.4给出。

对于参数 $w_K$，迭代式如下：

$$
w_K^{(k+1)} = w_q^{k} + \lambda*\frac{\partial L^1(w)}{\partial w_K^{k}}
\tag{5.7}
$$
>$\lambda$ 是学习率，$\frac{\partial L^1(w)}{\partial w_q^{k}}$ 由公式5.5给出。