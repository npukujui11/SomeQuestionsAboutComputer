## 涉及专业名词及前置知识
* **服务质量管理**：QoS，服务质量。是网络满足业务质量要求的控制机制，它是一个端到端的过程，需要业务发起者到响应者之间所经历的网络各节点共同协作，以保证服务质量。

* **网络KPI**：网络关键性能指标，网络设备在运行的过程中会持续的产生大量的数据，例如告警,KPI,日志,MML等等。KPI是能够反映网络性能与设备运行状态的一类指标。对KPI进行检测，能够及时发现网络质量劣化风险。目前KPI异常检测指的是通过算法分析KPI的时间序列数据，判断其是否出现异常行为。

* **CDN**：Content Delivery Network，内容分发网络是建立并覆盖在承载网之上，由分布在不同区域的边缘节点服务器群组成的分布式网络。CDN应用广泛，支持多种行业、多种场景内容加速，例如：图片小文件、大文件下载、视音频点播、直播流媒体、全站加速、安全加速。CDN运营商通常会收集每个网站的各种KPI，如流量、延迟、命中率等，并对这些多变量KPI进行异常检测，以检测业务故障或降级。

* **概率混合模型**：概率混合模型可以简单的理解为有多个（甚至是无数个）独立概率模型的凸组合(Convex Combination)，由于概率混合模型使用多个独立的概率分布，它可以描述一个复杂的数据分布，无论数据分布的结构如何复杂，总可以通过增加成分的方式来描述数据分布的局部特性，因此概率混合模型成为最有效的密度工具以及最常用的聚类工具之一。

* **变分自编码器**[<sup>[1]</sup>](#refer-anchor-1)： (Variation Auto-Encoders, VAE)是一种基于变分贝叶斯推断的生成式网络，它通过潜在随机变量（Latent Random Variables）来实现样本的生成，从而有更好的鲁棒性。

	+ 传统的**自编码器模型**是一种人工神经网络，用于学习未标记数据的有效编码（无监督学习）[<sup>[2]</sup>](#refer-anchor-2)。其两个主要应用是降维和信息检索[<sup>[3]</sup>](#refer-anchor-3)。其主要由两部分构成：编码器(encoder)和解码器（decoder）。
		
		- 如下图所示	<div><div align=center><img src="picture/自编码模型.png" alt="No Picture" style="zoom:100%"/><center><p>传统的自编码模型</p></center></div></div>
		
		-  解码消息的空间$\mathcal{X}$；编码信息的空间$\mathcal{Z}$。其中$\mathcal{X}$和$\mathcal{Z}$是欧几里得空间。其中$\mathcal{X} = \mathbb{R}^m, \mathcal{Z} =  \mathbb{R}^n$，其中$m, n$表示维数。

		- 两个参数化的函数：编码器(encoder)：$E_\phi:\mathcal{X}_\theta\rightarrow\mathcal{Z}$，参数为$\phi$；解码器(decoder)：$D_\theta:\mathcal{Z}\rightarrow\mathcal{X^{\prime}}$，参数为$\theta$；

		- ${\forall}x\in\mathcal{X}$写作$z=E_\phi(x)$，其中$z$称为潜在变量(the latent variable)。对于${\forall}z\in\mathcal{Z}$写作$x^{\prime}=D_\theta(z)$，一般称之为消息(message)

		- 通常，编码器和解码器都被定义为多层感知器[<sup>[4]</sup>](#refer-anchor-4)(Multilayer Perceptrons)。例如，一层MLP编码器$E_\phi$是$$E_\phi = \sigma(\mathcal{W}x+b)$$，其中$\sigma$是激活函数，$\mathcal{W}$是一个称为"权重(weight)"的矩阵，并且b是一个称为"偏差(bias)"的向量。

	+ 自动编码器的训练[<sup>[3]</sup>](#refer-anchor-3)。通过一个task去衡量模型的质量。设参考概率分布$\mu_{ref}$，对于$\forall{x}\in\mathcal{X}, x\sim\mu_{ref}$，因此可以把损失函数(loss function)定义为$${\displaystyle L(\theta ,\phi ) :=\mathbb{\mathbb {E} } _{x\sim\mu_{ref}}[d(x,D_{\theta}(E_{\phi}(X)))]}$$，对于给定任务的最佳自动编码器${\displaystyle (\mu _{ref},d)}$，其最优参数通过下式求解，$${\displaystyle \arg \min _{\theta ,\phi }L(\theta ,\phi )}$$

		- 对于最优自动编码器的搜索可以通过任何数学优化技术来完成，但是通常通过**梯度下降**。在大多数情况下参考概率分布为，$${\displaystyle \mu _{ref}={\frac {1}{N}}\sum _{i=1}^{N}\delta _{x_{i}}}$$

		- 质量函数是L2损失[<sup>[7]</sup>](#refer-anchor-7)：${\displaystyle d(x,x')=\|x-x^{\prime}\|_{2}^{2}}$。故寻找最优自编码器的问题是**最小二乘优化**[<sup>[5]</sup>](#refer-anchor-5) [<sup>[6]</sup>](#refer-anchor-6)：$${\displaystyle \min _{\theta ,\phi }L(\theta ,\phi ),{\text{where }}L(\theta ,\phi)={\frac {1}{N}}\sum _{i=1}^{N}\|x_{i}-D_{\theta }(E_{\phi }(x_{i}))\|_{2}^{2}}$$

	+ 变分自编码器构造依据的原理，具体结构如下
		- 如下图所示，与自动编码器由编码器与解码器两部分构成相似，VAE利用两个神经网络建立两个概率密度分布模型：一个用于原始输入数据的变分推断，生成隐变量的变分概率分布，称为**推断网络**；另一个根据生成的隐变量变分概率分布，还原生成原始数据的近似概率分布，称为**生成网络**。
		- <div><div align=center><img src="picture/变分自编码器模型.png" alt="No Picture" style="zoom:100%"/><center><p>变分自编码器结构</p></center></div></div>
		- 假设原始数据集为$$X = \{{x_i}\}_{i=1}^N$$
		- 每个数据样本$x_i$都是随机产生的相互独立、连续或离散的分布变量，生成的数据集合为$$X^{\prime} = \{{x_i}^{\prime}\}_{i=1}^N$$
	
		- 并且假设该过程产生隐变量$Z$，即$Z$是决定$X$属性的神秘原因(特征)。其中可观测变量$X$是一个高维空间的随机向量，不可观测变量$Z$是一个相对低维空间的随机向量，该生成模型可以分成两个过程：
			+ （1）隐变量$Z$后验分布的近似推断过程：$$q_{\phi}(z|x)$$，即推断网络。
			+ （2）生成变量$X^{\prime}$的条件分布生成过程：$$P_{\phi}(z)P_{\theta}(x^{\prime}|z)$$，即生成网络。
		
		- VAE的“编码器”和“解码器”的输出都是受参数约束变量的概率密度分布，而不是某种特定的编码。
		
		- 

	+ VAE和AE的差距[<sup>[8]</sup>](#refer-anchor-8)在于
		- AE的特点是数据相关的(data-specific)，这意味者自动编码器只能压缩那些与训练数据类似的数据，其是一类数据对应一种编码器，无法拓展一种编码器去应用于另一类数据。

		- 自动编码器是有损的，即解压缩的输出于原来的输入相比是退化的，MP3，JPEG等压缩算法也是如此。

		- VAE倾向于数据生成(data-generation)。只要训练好了decoder，我们就可以从某一个标准正态分布（一个区间）生成数据作为解码器decoder的输入，来生成类似的、但不完全相同于训练数据的新数据，也许是我们从类见过的新数据，作用类似于GAN。

		- 二者虽然都是$\mathcal{X}\rightarrow\mathcal{Z}\rightarrow\mathcal{X^{\prime}}$，但是AE寻找的是单值映射关系，即$z=f(x)$。
		
		- 而VAE寻找的是分布的映射关系，即$\mathcal{X}\rightarrow\mathcal{Z}$.
		
		- 为什么会有这区别呢？[<sup>[9]</sup>](#refer-anchor-9)AE的decoder做的是$\mathcal{Z}\rightarrow\mathcal{X^{\prime}}$变换，那么理论上它可以作为生成器使用。但这里有个问题，显然不是所有的$\mathbb{R}^{n}$都是有效的$\mathcal{Z}$。$\mathcal{Z}$的边界在哪里？如何得到有效的$\mathcal{Z}$，从而生成$\mathcal{x}$？这些都不是AE可以解决的，为了解决这个局限性，VAE映射的是分布，而分布可以通过采样得到有效的得到$\mathcal{z}$，从而生成相应的$\mathcal{x^{\prime}}$。

		- 变分自编码器通常与自编码器模型相关联，因为它具有架构亲和力，但是在目标和数学公式方面存在显著差异。变分自动编码器允许将统计推断问题（例如从另一个随机变量推断一个随机变量的值）重写为统计优化问题（即找到某些目标函数最小化的参数值）[<sup>[10]</sup>](#refer-anchor-10)[<sup>[11]</sup>](#refer-anchor-11)[<sup>[12]</sup>](#refer-anchor-12)。

## 前人工作成就总结

### 前人工作

一些无监督方法，利用循环神经网络RNN进行时间序列特征提取，用于多变量时间序列异常检测等等。

### 存在的问题
* **问题一**：单个网站在不同时间段的非平稳依赖（The non-stationary dependencies）会降低深度异常检测模型的性能:：

	+ 由于用户的正常行为或CDN的调度等，相应的KPI通常表现出非平稳的时间特征，不应将其归类为服务失败或退化。但是，这些类型的预期模式很难被目前的方法捕获，这会使得目前的模型在CDN KPI异常检测方面的表现性能很差。
	
	+ 从Figure1(a)和Figure1(d)可以明显看出，工作日和周末的用户请求行为是不同的。前一个网站显示了在周末相对于工作日的用户请求激增，而后一个数据显示了相反的情况。

<div align=center>
<img src="picture/figure1.png"
    alt="No Picture"
    style="zoom:100%"/>
<center><p>Figure1</p></center>
</div>

* **问题二**：不同的网站在CDN KPI中表现出不同的特征，但其中又有一些相似特征。这使得现有的**单个深度异常检测模型无法良好的捕捉这种动态复杂性**。
	+ Figure1(b)是另一种典型情况，即部分用户被CDN的调度中心调度到另一组边缘节点上；所以，KPI的变化时刻，发生在这段时间内。

	+ Figure1(a)中的视频点播网站的KPI通常与Figure1(e)中的直播网站有很大的不同。
	 	
		- 因为一般商业CDN会为数百甚至数千个网站提供服务，这些网站会因为服务类型和用户的请求行为的不同而表现出不同的特征，因此可以观察到不同网站的KPI在空间和时间特征上的变化。

* **问题三**：资源浪费
	+ 为了对多个网站进行有效的异常检测，现有的深度异常检测方法通常为每个网站训练一个单独的模型，从而存在为每个网站训练和维护大量的单独模型的问题，不仅消耗巨大的计算和存储资源，但也增加了模型维护的成本。
		- 例如，Figure1(c)和Figure1(f)显示出相似的特征，但它们属于不同网站的KPI。在这种情况下，为每个网站训练一个单独的模型完全是浪费。

## 难点与挑战

* Challenges:  In the scenario of Content Delivery Networks (CDN), KPIs that belong to diverse websites usually exhibit various structures at different timesteps and show the non-stationary sequential relationship between them, which is extremely difficult for the existing deep learning approaches to characterize and identify anomalies.

### 挑战

* 现有的许多深度学习方法<font color=red>（这里要细化）</font> ，模型学习正常情况下的网络KPI，以构建并训练模型，从而应用于无监督的网络KPI异常检测，即深度异常检测。

* **CDN KPI数据存在的问题**：
	+ CDN下具有多个节点，每个节点网络KPI曲线的多样性，即KPI曲线表现为周期型的，有稳定型的，也有不稳定型的。
	+ CDN下往往具备大量的节点网络，这些网络的异常种类多，核心网网元数据多，故障发生的类型也多种多样，导致了异常种类也多种多样，即KPI之间呈现出非平稳的顺序关系。

## 文章详细综述

### 作者的工作
* 表征不同的复杂KPI时间序列结构和动态特征；

* 基于以上问题，现有的深度学习方法非常难以表示和识别CDN网下节点网络的异常，因此作者提出了一种方法，一种适用于多变量CDN KPI的切换高斯混合变分循环神经网络(SGmVRN)。
	+ SGmVRNN引入变分循环结构，并将其潜在变量分配到混合高斯分布中，以对复杂的KPI时间序列进行建模并捕捉其中的不同结构和动态特征，而在下一步中，它结合了一种切换机制来表征这些多样性，从而学习更丰富的KPI表示。
	+ 为了有效的推理，我们开发了一种向上向下的自动编码推理方法，它结合了参数的自下而上的似然性和自下而上的先验信息，以实现准确的后验近似。

* **结果**：通过SGmVRNN方法构造出来的模型应用到不同网站的CDN KPI异常识别，SGmVRNN模型的平衡F分数明显高于目前最优秀的模型。


### 工作细节

#### 问题定义

第n个多元CDN KPI为$x_n={x_{1,n},x_{2,n},…,x_{T,n}}$,

* 其中$n=1,…,N$是KPI时间序列个数；
* T是$x_n$的持续时间；
* $x_{t,n}$是在时间T的观测值；
* $x_{t,n}∈\mathbb{R}^V$, $V$表示KPI的个数；
* $x_n∈\mathbb{R}^{T×V}$。

**多元CDN KPI的异常检测问题**被定义为确定某一网站在某一时间，$x_{t,n}$的观察结果是否异常的问题。

#### 模型阐述
 
<div align=center>
<img src="picture/figure2.png"
    alt="No Picture"
    style="zoom:100%"/>
<center><p>Figure2</p></center>
</div>

* 主要包含三个关键模块：
	+ **数据预处理模块(Data Preprocessing)**：负责对原始多元CDN KPI数据进行预处理。采用归一化和滑动时间窗方法
	+ **表示模块(Representation Model)**：使用SGmVRNN来学习多元CDN KPI的复杂结构和动态特性。
	+ **异常检测模块(Anomaly Detection)**：根据表示模型推导的重构概率来检测异常。

#### Data Preprocessing

#### Representation Model

##### SGmVRNN介绍

* SGmVRNN：将概率混合和切换机制融合到VRNN中，它将**概率混合**和**切换机制**融合到VRNN中，从而有效地模拟单个网站的多元CDN KPI相邻时间步之间的非平稳时间依赖性，以及不同网站之间的动态特性。
	+ SGmVRNN特点：能够在不同的时间学习不同的结构特征，并捕捉它们之间的各种时间依赖性，以表征不同CDN网站的多元KPI中的复杂结构和动态特征。
	+ 切换机制：
<div align=center>
<img src="picture/figure3.png"
    alt="No Picture"
    style="zoom:60%"/>
<center><p>Figure3</p></center>
</div>

* 如何解决主要挑战？
	+ 将切换机制与混合模型相结合，SGmVRNN可以利用切换机制对电流输入的刻画和多种时间变化的传输提供足够的表示能力，从而解决第一个问题。
	+ 利用混合高斯分布潜变量很好地处理多样化的网站，从而解决第二个问题。

##### VRNN(变分递归神经网络)

* 在每个时间步上都包含一个变分自编码器(VAE),假设每个时间步的潜随机变量来自高斯混合分布，其每个组件的参数由两个元素组成:
	+ 当前输入的特定先验
	+ 来自上一个时间步的潜在状态的切换。、

+ VRNN的潜随机变量先验，服从标准正态分布。
+ 引入一个离散指标变量c_(t,n)来指导当前时间步长的先验选择以及相邻时间步长之间信息的转换。

##### 模型参数学习

* 作者提出了一种自下而上自编码变分推理方法以获取SGmVRNN模型的参数，结合参数的自底向上似然和自底向上先验信息进行精确的后验逼近，从而得到SGmVRNN的丰富潜在表示。作者为了实现潜变量后验的精确逼近，我们提出了一种SGmVRNN的向上向下自编码变分推理方法。

## Reference
<div id="refer-anchor-1"></div>

- [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).`

<div id="refer-anchor-2"></div>

- [2] Kramer M A. Nonlinear principal component analysis using autoassociative neural networks[J]. AIChE journal, 1991, 37(2): 233-243.

<div id="refer-anchor-3"></div>

- [3] Goodfellow I, Bengio Y, Courville A. Deep learning[M]. MIT press, 2016.

<div id="refer-anchor-4"></div>

- [4] Hastie, Trevor, et al. The elements of statistical learning: data mining, inference, and prediction. Vol. 2. New York: springer, 2009.

<div id="refer-anchor-5"></div>

- [5] [Least squares optimization](https://people.duke.edu/~ccc14/sta-663-2018/notebooks/S09F_Least_Squares_Optimization.html#Least-squares-optimization)

<div id="refer-anchor-6"></div>

- [6] Grisetti G, Guadagnino T, Aloise I, et al. Least squares optimization: From theory to practice[J]. Robotics, 2020, 9(3): 51.

<div id="refer-anchor-7"></div>

- [7] Bühlmann, Peter, and Bin Yu. "Boosting with the L 2 loss: regression and classification." Journal of the American Statistical Association 98.462 (2003): 324-339.

<div id="refer-anchor-8"></div>

- [8] An, Jinwon, and Sungzoon Cho. "Variational autoencoder based anomaly detection using reconstruction probability." Special Lecture on IE 2.1 (2015): 1-18.

<div id="refer-anchor-9"></div>

- [9] [【深度学习】AE与VAE] (https://blog.csdn.net/sinat_36197913/article/details/93630246?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-93630246-blog-110879937.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-93630246-blog-110879937.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2)

<div id="refer-anchor-10"></div>

- [10] Kramer, Mark A. "Nonlinear principal component analysis using autoassociative neural networks." AIChE journal 37.2 (1991): 233-243.

<div id="refer-anchor-11"></div>

- [11] Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "Reducing the dimensionality of data with neural networks." science 313.5786 (2006): 504-507.

<div id="refer-anchor-12"></div>

- [12] Jang, E. "A beginner’s guide to variational methods: Mean-field approximation." (2016).