<font color=#ff0000>红色</font>

<font color="blue">蓝色</font>

<font color=Yellow>黄色</font>

<font color=YellowGreen>黄绿色</font>

<font face="黑体">黑体</font>

<font face="宋体">宋体</font>

|  **论文标题**   | **作者**  | **keyword** | **问题描述**| **难点与挑战** | **总结方法** | **性能分析** | **未来研究方向** | **研究侧重点** | **涉及概念** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| <font face="Times New Roman">New robust metric learning model using maximum correntropy criterion</font>  | <font face="Times New Roman">J. Xu, L. Luo, C. Deng and H. Huang</font> | <font face="Times New Roman">Robust Metric Learning,  Maximum Correntropy Criterion</font> | <font face="Times New Roman">Most existing metric learning methods aim to learn an optimal Mahalanobis distance matrix $M$, under which data samples from the same class are forced to be close to each other and those from different classes are pushed far away.  However, the traditional metric learning algorithms are not robust to achieve good performance when they are applied to the occlusion data, which often appear in image and video data mining applications.  </font> | <font face="Times New Roman">Now, various kinds of metric learning algorithms have been proposed in literature, e.g., large-margin nearest neighbors (LMNN), information-theoretic metric learning (ITML), logistic discriminant metric learning (LDML), etc. Although these metric learning methods were successful in solving many problems, they cannot handle the noisy data well, which has become the bottleneck for them to achieve good performance in real-world applications. Although notable improvements have been achieved by these methods, they ignore the effect of real-world malicious occlusions or corruptions on the model performance.</font> | <font face="Times New Roman">The author propose a new robust metric learning approach by introducing the maximum correntropy criterion to deal with real-world malicious occlusions or corruptions.</font> | <font face="Times New Roman">Leveraging the half-quadratic optimization technique, we derive an efficient algorithm to solve the proposed new model and provide its convergence guarantee as well. Extensive experiments on various occluded data sets indicate that our proposed model can achieve more promising performance than other related methods.</font> | $NaN$ | $NaN$ | <font face="Times New Roman">Mahalanobis distance, Maximum Correntropy Criterion, the half-quadratic optimization technique</font> | 
| $NaN$ | $NaN$ | **鲁棒度量学习**，**最大相关熵准则** | 数据挖掘中常见的度量学习方法一般是为了学习一个最优的马氏距离矩阵M，即同一类的数据样本被迫彼此接近，不同类的数据样本间隔很远，但应用于遮挡数据时，鲁棒性不佳，无法取得良好的性能，这在图像和视频数据挖掘应用中经常出现。 | 目前的各种度量学习算法，如LMNN（大边界最近邻）、ITML（信息理论度量学习）、LDML（逻辑判别度量学习）等。虽然这些度量学习方法成功地解决了很多问题，但它们不能很好地处理噪声数据，这已经成为它们在现实应用中取得良好性能的瓶颈。尽管这些方法已经取得了显著的改进，但它们忽略了现实世界中恶意遮挡或破坏对模型性能的影响。 | 作者提出了一种新的鲁棒度量学习方法，通过引入最大相关熵准则来处理现实世界中的恶意遮挡或破坏。 | 利用半二次优化技术，得到了求解新模型的有效算法，并提供了收敛保证。在各种闭塞数据集上的大量实验表明，该模型比其他相关方法具有更好的性能。| $NaN$ |$NaN$ | **马氏距离**、**最大相关熵准则**、**半二次优化技术** |

