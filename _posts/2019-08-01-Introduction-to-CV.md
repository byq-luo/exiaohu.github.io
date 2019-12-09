---
layout: mypost
title: 卷积神经网络及其在计算机视觉方面的应用整理
categories: [Convolutional Neural Networks, Computer Vision]
---

# Convolution

## Discrete convolutions
- 多维数据
- 若干个维度是有序的（图片的宽和高，音视频的时间轴）
- channels：used to access different views of the data （图片、视频的红绿蓝通道，声音的左右声道）
- 特性：
  - 参数共享
  - 平移等变
  - 稀疏链接
- 参数
  - 输入size
  - 输入、输出通道数
  - stride，大于1时为降采样
  - padding
    - zero padding, average padding, nearest padding
    - half padding: 让输出feature map和输入同样大小
    - full padding: 考虑到输入feature mao上的每一个可能的部分或全部卷积操作，逐渐增加feature map
  - 输入size `i`, 卷积核大小`k`, padding `p`和 stride `s`,输出size `o` 为：
    ```math
    o = \left\lfloor \frac{ i + 2p - k}{s} \right\rfloor + 1
    ```


## Pooling
- splitting the input in (usually non-overlapping) patches and outputting the pooling value of each patch
- 参数
  - 输入size
  - pooling window size 池化窗口大小
  - stride 步长
  - 输入size `i`, 卷积核大小`k` 和 stride `s`,输出size `o` 为：
    ```math
    o = \left\lfloor \frac{ i - k}{s} \right\rfloor + 1
    ```

## Transposed convolution
- alias: fractionally strided convolutions, deconvolutions
- a transformation going in the opposite direction of a normal convolution
- convolution with padding vs. transposed convolution
  - adding many columns and rows of zeros to the input, resulting in a much less efficient implementation

## Dilated/Atrous convolutions
- “inflate” the kernel by inserting spaces between the kernel elements
- 额外的参数`dilation rate`
- cheaply increase the receptive field of output units without increasing the kernel size
- dilated kernel size:
    ```math
    \hat{s} = k + (k - 1)(d - 1)
    ```
- 输入size `i`, 卷积核大小`k`, padding `p`和 stride `s`, dilation rate `d`, 输出size `o` 为：
    ```math
    o = \left\lfloor \frac{ i + 2p - k - (k - 1)(d - 1)}{s} \right\rfloor + 1
    ```


# Classification

## LeNet-5 (Gradient-Based Learning Applied to Document Recognition)
- for handwritten character recognition
- input `32`x`32` pixel image;
- seven hidden layers(C-Convolutional, S-Subsample, F-Fully Connected);
  - C1: `6` feature maps, `5`x`5`, output `28`x`28`, `156` trainable parameters.
  - S2: `6` feature maps, `2`x`2`, output `14`x`14`, `sigmoid` activation function, `12` trainable parameters, `4` units added then multiplied by `1` trainable coefficient, and then added to `1` trainable bias.
  - C3: `16` feature maps, `5`x`5`, output `10`x`10`, `1516` trainable parameters. The first `6` C3 feature maps take inputs from every contiguous subsets of `3` feature maps in S2. The next `6` take input from every contiguous subset of `4`. The next `3` take input from some discontinuous subsets of `4`. Finally, the last `1` takes input from all `6` S2 feature maps.
  - S4: `16` feature maps, `2`x`2`, output `5`x`5`, sigmoid activation, `32` trainable parameters.
  - C5: `120` feature maps, `5`x`5`, output `1`x`1`, `48120` trainable connections.
  - F6: `84` units, `tanh` activation function, `10164` trainable parameters.
  - Output Layer, RBF (Radial Basis Function). 虽是全连接，但参数W是给定值。输入的84维向量相当于12∗7的比特图。输出的每一个值，代表了输入与输出对应的参数权重的均方误差MSE。
- downsampling: 
```math
sigmoid(w*(\sum_{i=0}^1\sum_{j=0}^1 x_{ij})+b)
```
where `w` and `b` is trainable parameters.
  
## AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)
- for Image Classification
- input `256`x`256` pixel image
- `5` convolutional and `3` fully connected layers
- `ReLU` activation
- 参数`60 Million`，MACs `1.1 Billion` （实际计算量比这个值小，因为Conv层中使用了group）
- 第一次引入 `ReLU` ，并使用了 `overlapping Max Pooling`
在前两个全连接层使用了系数为`0.5`的 `Dropout`，因此测试时需要将结果乘以`0.5`
- 在论文中，还引入了局部响应归一化层`LRN`。但后来证明没有效果提升。同时，`overlapping Max Pooling`也没有被广泛应用。
- 数据增强
  - 对于训练集，随机剪裁。对于测试集，将原始图片和对应的水平镜像从中间和四边剪切，然后将这十个预测结果取平均
  - PCA jittering。基于整个训练集特征值 `λ` 和特征向量 `P`，对于每个epoch的每一张图像，从均值`0`、方差`0.1`的高斯分布随机抽取 `α`。对原始图片三维通道做 `λ * α * P` 映射。


## ZFNet (Visualizing and Understanding Convolutional Networks)
- 引入 DeconvNet 来可视化某一 `feature map`，将该层其余 `feature maps` 设置为`0`，然后经过一系列的(i) `unpool` (ii) `rectify` (iii) `filters` 映射回像素级别。其中：
  1. `unpool`：`max-pooling`同时记录位置信息。
  2. `ReLU`。
  3. 原卷积对应矩阵的转置。
- 论文中使用可视化方法：
  - 对于某一层的某个 `feature map`，我们在验证集中寻找使该 `feature map` 的 `response` 最大的九张图片，画出这九张图片中的该 feature map 反卷积映射的结果，并和原图相对应的 patch 对比
  - 特征可视化：层数越高，提取的特征越复杂，不变性越明显，越 `discrimination`。
- 越低层的神经网络参数，收敛越早
- 特征不变性：
  1. 图像缩放、平移对模型第一层影响较大，对后面基本没有影响；
  2. 图像旋转后，特征不具有不变性。
- 通过第一层和第二层可视化对AlexNet进行改造得到ZFNet：减小感受野，降低步长，有利于在第一层学到各种频段的图像特征。
- 模型对局部特征敏感
- 模型具有特征泛化性，因此可以用于迁移学习。


## NIN (Network In Network)
- replace `convolutional filter` and `activation function` with a general nonlinear function approximator, i.e. multilayer perceptron.
- replace the `fully connected` in output layer with `average pooling` for classification.
- 引入`1`x`1`卷积
  - Original：Conv `3`x`3` -> `ReLU` -> `Pooling`
  - MLPConv: Conv `3`x`3` -> `ReLU` -> Conv `1`x`1` -> `ReLU` -> ... -> `Pooling`


## VGG (Very Deep Convolutional Networks for Large-Scale Image Recognition)
- 使用了统一的卷积结构，证明了深度对模型效果的影响。`LRN`(局部响应归一化)层没有提升效果。
- 堆叠多个`3`x`3`的感受野，可以获得类似于更大感受野的效果。同时，多层`3`x`3`卷积堆叠对应的参数更少（减少参数相当于正则化效果）
- 运用了 Network in Network 中提出的`1`x`1`卷积。
- 训练方式： `256` `batch size`，`0.9` `momentum`，`5e-4` `weight decay`，`0.5` `dropout ratio`。`learning rate`：初始`1e-2`，每当停止提升后下降为之前的十分之一
- 数据增强
  - 颜色增强
    1. `color jittering`: HSV颜色空间随机改变图像原有的饱和度和明度（即，改变 S 和 V 通道的值）或对色调(Hue)进行小范围微调。
    2. `PCA jittering`: 首先按照RGB三个颜色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值，用来做`PCA Jittering`
  - 尺度变换：训练集随机缩放到[`256`, `512`]，然后随机剪切到`224`x`224`
  - 尺度变换对应的测试方法：
    1. 随机裁剪，取平均，类似AlexNet
    2. 将FC转为Conv，原始图片直接输入模型，这时输出不再是`1`x`1`x`1000`，而是`N`x`N`x`1000`，然后取平均。

## GoogLeNet, Inceptionv2, Inceptionv3
### GoogLeNet (Going deeper with convolutions)
- 论文发表之前相关的工作：当时研究者关注增加层数和filter数目（可能会导致小样本过拟合，并需要大量的计算资源），并通过Dropout防止过拟合。尽管有人认为 Max Pooling 造成了空间信息的损失，但这种结构在 localization、detection、human pose estimation 中均取得很好的成绩。
- Network-In-Network 中引入1x1卷积，增加了神经网络的表示能力【representational power】。GoogLeNet中的 Inception 结构也运用了1x1卷积来进行降维处理。
- 为解决过拟合和计算代价高的问题，使用稀疏网络来代替全连接网络。在实际中，即使用卷积层。
Inception结构：对于输入的 feature maps，分别通过1x1卷积、3x3卷积、5x5卷积和 Max-Pooling 层，并将输出的 feature maps 连接起来作为 Inception 的输出【同时获得不同感受野的信息】。在3x3卷积、5x5卷积前面和池化层后面接1x1卷积，起降维的作用。
- While most current vision oriented ma-
chine learning systems utilize sparsity in the spatial domain
just by the virtue of employing convolutions, they make use of `filter-level sparsity`.
- judiciously reducing dimension wherever the computational requirements would increase too much otherwise.
- 1×1 convolutions are used to compute reductions before the expensive 3×3 and 5×5 convolutions.
- visual information should be processed at various scales and then aggregated so that
the next stage can abstract features from the different scales simultaneously.
### Inceptionv2, Inceptionv3 (Rethinking the Inception Architecture for Computer Vision)
- Principles:
  - Avoid representational bottlenecks, especially early in the network. Feed-forward networks can be represented by an acyclic graph from the input layer(s) to the classifier or regressor. This defines a clear direction for the information flow. For any cut separating the inputs from the outputs, one can access the amount of information passing though the cut. One should avoid bottlenecks with extreme compression. In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand. Theoretically, information content can not be assessed merely by the dimensionality of the representation as it discards important factors like correlation structure; the dimensionality merely provides a rough estimate of information content.
  - Higher dimensional representations are easier to process locally within a network. Increasing the activations per tile in a convolutional network allows for more disentangled features. The resulting networks will train faster.
  - Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. For example, before performing a more spread out (e.g. 3 × 3) convolution, one can reduce the dimension of the input representation before the spatial aggregation without expecting serious adverse effects. We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension reduction, if the outputs are used in a spatial aggregation context. Given that these signals should be easily compressible, the dimension reduction even promotes faster learning.
  - Balance the width and depth of the network. Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. Increasing both the width and the depth of the network can contribute to higher quality networks. However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel. The computational budget should therefore be distributed in a balanced way between the depth and width of the network.
- a `5` × `5` filter can capture dependencies between signals between activations of units further away in the earlier layers, so a reduction of the geometric size of the filters comes at a large cost of expressiveness.
- Auxiliary Classifiers:
  - push useful gradients to the lower layers to make them immediately useful and improve the convergence during training by combating the vanishing gradient problem in very deep networks
  - Near the end of training, the network with the auxiliary branches starts to overtake the accuracy of the network without any auxiliary branch and reaches a slightly higher plateau.
- 5x5卷积的感受野与两个3x3卷积堆叠所对应的感受野相同。使用后者可以大大减少网络参数。7x7同理。此外，两个3x3卷积后各连接一个非线性层的效果优于仅在最后连接一个非线性层
- NxN的卷积可以用1xN与Nx1的卷积堆叠实现。
- Label Smoothing:
  - For a training example with ground-truth label y, we replace the label distribution `q(k|x) = \delta_{k,y}` with
  ```math
  q'(k|x) = (1-\epsilon)\delta_{k, y} + \epsilon u(k)
  ```
  where `u(k)` is a prior distribution over labels, independent of the training example `x`.

## ResNet (Deep Residual Learning for Image Recognition)
- 退化问题: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.
- if the added layers can be constructed as identity mappings, a deeper model should
have training error no greater than its shallower counterpart.
- When the dimensions increase, two options are conisdered: 
  1. The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; 
  2. The projection shortcut with an `affine transformation` is used to match dimensions (done by 1×1 convolutions)
- 通常，一个 `residual unit` 的残差部分使用二至三层的函数映射（或称卷积层），`shortcut` 部分与残差部分进行 `eltwise add` 后再连接非线性层。

## DenseNet (Densely Connected Convolutional Networks)
- connects each layer to every other layer in a feed-forward fashion
- DenseNet 极大地增加了特征重用的能力，其有以下优点。 
  1. 参数少，通过向后连接的方式保留学到的信息；(very narrow)
  2. 改进了前向、反向传播，更易训练；
  3. 增加了监督学习的能力；
  4. 在小数据上不易过拟合，即增加了正则化的能力。
- Dense Block 中，对于任意一层的 feature maps，一方面会通过 BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)得到下一层的 feature maps，另一方面会与其后的每一层 feature maps 连接在一起。并提出了 growth rate 的概念，增长率 k 是3x3卷积层输出的 feature maps 数，而1x1卷积层输出的 feature maps 数为 4k
- 在 ImageNet 比赛中运用的模型，每两个 Dense Block 中间以 Batch Normalization - ReLU - Conv(1x1) - Aver Pooling(2x2) 相连。
- combine features by concatenating them
- The global state, once written, can be accessed from everywhere within the network and, unlike in traditional network architectures, there is no need to replicate it from layer to layer.


# Object Detection
## Selective Search (Selective Search for Object Recognition)
- Exhaustive Search: 择一个窗口（window）扫描整张图像（image），改变窗口的大小，继续扫描整张图像。
- 在做物体识别（Object Recognition）过程中，不能通过单一的策略来区分不同的物体，需要充分考虑图像物体的多样性（diversity）。另外，在图像中物体的布局有一定的层次（hierarchical）关系，考虑这种关系才能够更好地对物体的类别（category）进行区分。
  1. 穷举搜索（Exhaustive Selective）通过改变窗口大小来适应物体的不同尺度，选择搜索（Selective Search）同样无法避免这个问题。算法采用了图像分割（Image Segmentation）以及使用一种层次算法（Hierarchical Algorithm）有效地解决了这个问题;
  2. 多样化（Diversification）：单一的策略无法应对多种类别的图像。使用颜色（color）、纹理（texture）、大小（size）等多种策略对（【1】中分割好的）区域（region）进行合并。
- 算法思路；
  1. Obtain initial regions R = {r 1 , · · · , r n } using Felzenszwalb and
Huttenlocher (2004)
  2. 计算相邻区域间的相似度，组成集合`S`
  3. 迭代执行，直到只剩下一个区域：
     1. 找到相似度最大的两个区域；
     2. 从集合`S`中删除与这两个区域相关的相似度；
     3. 合并这两个区域，并计算合并后所得新区域与与其相邻区域的相似度，加入到集合`S`中。
- 多样化：
  - by using a variety of colour spaces with: 
    1. RGB, 
    2. the intensity (grey-scale image) I,
    3. Lab,
    4. the rg channels of normalized RGB plus intensity denoted as rg I,
    5. HSV,
    6. normalized RGB denoted as rgb,
    7. C Geusebroek et al. (2001) which is an opponent colour space where intensity is divided out,
    8. the Hue channel H from HSV.
  - by using different similarity measures `s_{ij}`: 
    1. colour similarity: for each region we obtain one-dimensional colour histograms for each colour channel using 25 bins.
    2. texture similarity: represent texture using fast SIFT-like measurements.
    3. encourages small regions to merge early. the size of the image in pixels.
    4. The idea is to fill gaps: if `r_i` is contained in `r_j` it is logical to merge these first in order to avoid any holes. On the other hand, if `r_i` and `r_j` are hardly touching each other they will likely form a strange region and should not be merged. To keep the measure fast, we use only the size of the regions and of the containing boxes. 
  - by varying our starting regions: different starting regions are (already) obtained by varying the colour spaces, each which has different invariance properties.


## SPP-net (Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)
- problem: Existing deep convolutional neural networks (CNNs) require a fixed-size input image. This requirement is “artificial” and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale.
- traditional approaches:
  - crop: 截取原图片的一个固定大小的patch, 物体可能会产生截断，尤其是长宽比大的图片.
  - wrap: 将原图片缩放到一个固定大小的patch, 物体被拉伸，失去“原形”，尤其是长宽比大的图片
- idea: CNN的卷积层可以处理任意尺度的输入，全连接层处限制尺度——如果找到一个方法，在全连接层之前将其输入限制到等长即可。
- 方法: 将原来固定大小的窗口的最后一层池化层改成自适应窗口大小，窗口的大小和 activation map 成比例，保证经过pooling后出来的feature的长度一致.


## R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN
### R-CNN (Rich feature hierarchies for accurate object detection and semantic segmentation)
- insights:
  - one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects;
  - when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost
- Object detection system overview:
  1. takes an input image;
  2. extracts around 2000 bottom-up region proposals;
  3. computes features for each proposal using a large convolutional neural network (CNN);
  4. classifies each region using class-specific linear SVMs.
- localization approach:
  1. frames localization as a regression problem
  2. build a sliding-window detector
  3. proposal-classification framework: by operating within the “recognition using regions” paradigm. generates category-independent region proposals for the input image, extracts a fixed-length feature vector from each proposal using a CNN, and then classifies each region with category-specific linear SVMs.
- labeled data is scarce and the amount currently available is insufficient for training a large CNN:
  1. unsupervised pre-training followed by supervised fine-tuning
  2. supervised pre-training followed by domain-specific fine-tuning
- modules:
  1. generates category-independent region proposals:
     - use selective search
  2. a large convolutional neural network that extracts a fixed-length feature vector from each region:
     - Regardless of the size or aspect ratio of the candidate region, we warp all pixels in a tight bounding box around it to the required size. Prior to warping, we dilate the tight bounding box so that at the warped size there are exactly `p` pixels of warped image context around the original box (we use `p = 16`).
     - pretrained AlexNet
  3. a set of classspecific linear SVMs:
- questions:
  1. 为什么不直接用CNN分类？
     - cnn在训练的时候，对训练数据做了比较宽松的标注，比如一个 bounding box 可能只包含物体的一部分，那么我也把它标注为正样本，用于训练cnn；采用这个方法的主要原因在于因为CNN容易过拟合，所以需要大量的训练数据，所以在CNN训练阶段我们是对 Bounding box的位置限制条件限制的比较松(IOU只要大于0.5都被标注为正样本了)；
     - 然而svm训练的时候，因为svm适用于少样本训练，所以对于训练样本数据的IOU要求比较严格，我们只有当 bounding box 把整个物体都包含进去了，我们才把它标注为物体类别，然后训练svm。 
  2. 直接使用预训练好的参数不经过fine-tune可行吗？
     - 如果不针对特定任务进行 fine-tuning ，而是把CNN当做特征提取器，卷积层所学到的特征其实就是基础的共享特征提取层，就类似于 SIFT 算法一样，可以用于提取各种图片的特征，而f6、f7所学习到的特征是用于针对特定任务的特征。打个比方：对于人脸性别识别来说，一个CNN模型前面的卷积层所学习到的特征就类似于学习人脸共性特征，然后全连接层所学习的特征就是针对性别分类的特征了。

### Fast R-CNN (Fast R-CNN)
- drawbacks of R-CNN:
  - Training is a multi-stage pipeline.
  - Training is expensive in space and time.
  - Object detection is slow.
- contributions: 
  - Higher detection quality
  - Training is single-stage, using a multi-task loss
  - Training can update all network layers
  - No disk storage is required for feature caching
- architecture: An input image and multiple regions of interest (RoIs) are input into a fully convolutional network. Each RoI is pooled into a fixed-size feature map and then mapped to a feature vector by fully connected layers (FCs). The network has two output vectors per RoI: softmax probabilities and per-class bounding-box regression offsets. The architecture is trained end-to-end with a multi-task loss.
  - RoI pooling layer
    1. 将image中的roi定位到feature map中对应区域（patch）;
    2. convert the features inside any valid region of interest into a small feature map with a fixed spatial extent
- loss:
  - 分类LOSS, N+1，N个物体类别和背景，使用softmax分类器。
  - 回归loss，是一个4xN路输出的 bounding-box regressor (有4个元素(x,y,w,h)，左上角坐标(x,y)，宽w，高h)，也就是说对于每个类别都会训练一个单独的regressor。
- disscussions:
  1. 测试时速度
     - 测试时速度慢：R-CNN 把一张图像分解成大量的建议框，每个建议框拉伸形成的图像都会单独通过 CNN 提取特征.实际上这些建议框之间大量重叠，特征值之间完全可以共享，造成了运算能力的浪费.
     - FAST-RCNN 将整张图像归一化后直接送入 CNN，在最后的卷积层输出的feature map上，加入建议框信息，使得在此之前的CNN运算得以共享.
  2. 训练时速度
     - R-CNN在训练时，是在采用SVM分类之前，把通过CNN提取的特征存储在硬盘上.这种方法造成了训练性能低下，因为在硬盘上大量的读写数据会造成训练速度缓慢.
     - FAST-RCNN在训练时，只需要将一张图像送入网络，每张图像一次性地提取CNN特征和建议区域，训练数据在GPU内存里直接进Loss层，这样候选区域的前几层特征不需要再重复计算且不再需要把大量数据存储在硬盘上.
  3. 训练所需空间
     -  训练所需空间大：R-CNN 中独立的 SVM 分类器和回归器需要大量特征作为训练样本，需要大量的硬盘空间.
     - FAST-RCNN把类别判断和位置回归统一用深度网络实现，不再需要额外存储.
  
### Faster R-CNN (Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks)
- Region Proposal Network: simultaneously predicts object bounds and objectness scores at each position
- slide a small network over the conv feature map output by the last shared conv layer. This network is fully connected to an n × n spatial window of the input conv feature map. Each sliding window is mapped to a lower-dimensional vector. This vector is fed into two sibling fully-connected layers—a box-regression layer (reg) and a box-classification layer (cls). simultaneously predict `k` region proposals.
  - reg: has `4k` outputs encoding the coordinates of `k` boxes
  - cls: outputs `2k` scores that estimate probability of object/not-object for each proposal.
- difference with Fast R-CNN:
  - 使用RPN(Region Proposal Network)代替Selective Search方法产生建议窗口；
  - 产生建议窗口的CNN和目标检测的CNN共享
- archor: The `k` proposals are parameterized relative to `k` reference boxes, called anchors. Each anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio. We use `3` scales and `3` aspect ratios, yielding `k = 9` anchors at each sliding position.

### Mask R-CNN (Mask R-CNN)
- detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance
- extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.
- Objective:
  - 高速和高准确率：为了实现这个目的，作者选用了经典的目标检测算法Faster-rcnn和经典的语义分割算法FCN。Faster-rcnn可以既快又准的完成目标检测的功能；FCN可以精准的完成语义分割的功能，这两个算法都是对应领域中的经典之作。Mask R-CNN比Faster-rcnn复杂，但是最终仍然可以达到5fps的速度，这和原始的Faster-rcnn的速度相当。由于发现了ROI Pooling中所存在的像素偏差问题，提出了对应的ROIAlign策略，加上FCN精准的像素MASK，使得其可以获得高准确率。
  - 简单直观：整个Mask R-CNN算法的思路很简单，就是在原始Faster-rcnn算法的基础上面增加了FCN来产生对应的MASK分支。即Faster-rcnn + FCN，更细致的是 RPN + ROIAlign + Fast-rcnn + FCN。
  - 易于使用：整个Mask R-CNN算法非常的灵活，可以用来完成多种任务，包括目标分类、目标检测、语义分割、实例分割、人体姿态识别等多个任务，这将其易于使用的特点展现的淋漓尽致。我很少见到有哪个算法有这么好的扩展性和易用性，值得我们学习和借鉴。除此之外，我们可以更换不同的backbone architecture和Head Architecture来获得不同性能的结果。
- Architecture:
  - Faster R-CNN: 见上
  - FCN: 见上
  - ROI Align: from `original image space` to `feature map space` to `ROI space`, coordinates quantization (roundings) lead to inaccuracy. RoIAlign layer that removes the harsh quantization of RoIPool, properly aligning the extracted features with the input. Our proposed change is simple: we avoid any quantization of the RoI boundaries or bins. Instead, `bilinear interpolation` is used.


## SSD (SSD: Single Shot MultiBox Detector)
- detecting objects in images using a single deep neural network
- discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location


## YOLO, YOLO9000, YOLOv3
### YOLO (You Only Look Once: Unified, Real-Time Object Detection)
- a single neural network
- advantages:
  - extremely fast
  - less background errors tjam Fast R-CNN
  - highly generalizable than DPM and R-CNN
- disadvantages:
  - struggles with small objects that appear in groups, such as flocks of birds.
  - struggles to generalize to objects in new or unusual aspect ratios or configurations
  - loss function treats errors the same in small bounding boxes versus large bounding boxes.
  - more localization errors than faster R-CNN
  - relatively low recall compared to region proposal-based methods
- first, divides the input image into an `S` × `S` grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
- Each grid cell predicts `B` bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and
also how accurate it thinks the box is that it predicts. 
- Each **bounding box** consists of `5` predictions: `x`, `y`, `w`, `h`, and `confidence`. The `(x, y)` coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.
- Each **grid cell** also predicts `C` conditional class probabilities, `Pr(Classi
|Object)`. These probabilities are conditioned on the grid cell containing an object.


### YOLOv2, YOLO9000 (YOLO9000: Better, Faster, Stronger)
- improved YOLO (YOLOv2)
  - better
      - batch normalization
      - high resolution clasifier
      - convolutional with anchor boxes
      - dimension clusters
      - direct location prediction
      - fine-grained features
      - multi-scale training
    - faster
      - darknet
- multi-scale training method
- jointly train on object detection and classification
  - predict detections for object classes that don’t have labelled detection data
  - uses images labelled for detection to learn detection-specific information like bounding box coordinate prediction and objectness as well as how to classify common objects.
  - uses images with only class labels to expand the number of categories it can detect.
  - When the network sees an image labelled for detection we can backpropagate based on the full YOLOv2 loss function. 
  - When it sees a classification image we only backpropagate loss from the classification-specific parts of the architecture.
  - **Hierarchical classification**
    - building a hierarchical tree from the concepts in ImageNet
    - predict conditional probabilities at every node for the probability of each hyponym of that synset given that synset.


### YOLOv3 (YOLOv3: An Incremental Improvement)

# Image Segmentation
## FCN (Fully Convolutional Networks for Semantic Segmentation)
- convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation: (1) for pixelwise prediction (2) from supervised pre-training.
- ine-tune all layers by backpropagation through the whole net.
- transfers recent success in classification [19, 31, 32] to dense prediction by reinterpreting classification nets as fully convolutional and fine-tuning from
their learned representations.
  - In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling
  - define a novel “skip” architecture to combine deep, coarse, semantic information and shallow, fine, appearance information
- Transforming fully connected layers into convolution layers enables a classification net to output a heatmap.
- Upsampling is backwards strided convolution
  - interpolation
  - upsampling with factor `f` is convolution with a fractional input stride of `1/f`. So long as `f` is integral, a natural way to upsample is therefore backwards convolution (sometimes called deconvolution) with an output stride of `f`.
- weakness:
  - 分割结果不够精细。进行8倍上采样虽然比32倍的效果好了很多，但是上采样的结果还是比较模糊和平滑，对图像中的细节不敏感。
  - 忽略了在通常的基于像素分类的分割方法中使用的空间规整（spatial regularization）步骤，缺乏空间一致性。

## U-Net (U-Net: Convolutional Networks for Biomedical Image Segmentation)
![image.png](unet.png)
- it works with very few training images and yields more precise segmentations.
- a network architecture: consists of a contracting path (left side) and an expansive path (right side). 
  - The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. 
  - Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. 
  - At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.
- a training strategy: shift and rotation, random elastic deformations, Dropout.
- advanced U-Net
  - [The Importance of Skip Connections in Biomedical Image Segmentation](https://arxiv.org/abs/1608.04117)
    - swap out the basic stacked convolution blocks in favor of residual blocks
  - [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326)
    - dense blocks

## DeepLab, DeepLabv2, DeepLabv3, DeepLabv3+, Auto-DeepLab
### DeepLab (Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs)
- Composed of **DCNN**s and **CRF**s
- use pretrained VGG-16 as feature extractor
- convert the fully-connected layers of VGG-16 into convolutional ones and run the network in a convolutional fashion on the image at its original resolution
  - yields very sparsely computed detection scores (with a stride of 32 pixels)
    - skip subsampling after the last two max-pooling layers
    - modify the convolutional filters in the layers to Atrous Convolution
- use fully connected CRF model to refine the segmentation result
- not end-to-end
### DeepLabv2 (DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs)
![image.png](deeplabv2.png)
- **‘atrous convolution’**: allows to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks and to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation
- **atrous spatial pyramid pooling (ASSP)**: probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales
  - standard multiscale processing: extract DCNN score maps from multiple rescaled versions of the original image using parallel DCNN branches that share the same parameters (significantly performance improvement, at the cost of computing feature responses at all DCNN layers for multiple scales of input)
- **fully connected CRF**: combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance
- VGG-16 or ResNet-101
  - transforming all the fully connected layers to  convolutional layers
  - increasing feature resolution through atrous convolutional layers
- improvement after DeepLab
  - learning rate policy
  - ASSP
  - Deeper Networks and Multiscale Processing

### DeepLabv3 (Rethinking Atrous Convolution for Semantic Image Segmentation)
![image.png](deeplabv3.png)
- in cascade
  - duplicate several copies of the last ResNet block and arrange them in cascade
- in parallel
  - include batch normalization within ASPP
  - adopt image-level features
- apply `global average pooling` on the last feature map of the model, feed the resulting image-level features to a `1 × 1` convolution with `256` filters (and batch normalization), and then bilinearly upsample the feature to the desired spatial dimension. In the end, our improved ASPP consists of 
  - (a) one `1 × 1` convolution and three `3 × 3` convolutions with `rates = (6, 12, 18)` when `output stride = 16` (all with `256` filters and batch normalization),
  - (b) the image-level features.
- The resulting features from all the branches are then concatenated and pass through another `1 × 1` convolution (also with `256` filters and batch normalization) before the final `1 × 1` convolution which generates the final logits.

### DeepLabv3+ (Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation)
![image.png](deeplabv3plus1.png)
- combine `spatial pyramid pooling module` and `encode-decoder structure`
  - `spatial pyramid pooling module` encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view
    - it is computationally prohibitive to extract output feature maps that are 8, or even 4 times smaller than the input resolution
  - `encode-decoder structure` capture sharper object boundaries by gradually recovering the spatial information
- The rich semantic information is encoded in the output of DeepLabv3, with atrous convolution allowing one to control the den- sity of the encoder features, depending on the budget of computation resources. Furthermore, the decoder module allows detailed object boundary recovery.
![image.png](deeplabv3plus2.png)
- `depthwise separable convolution` or `group convolution`: a powerful operation to reduce the computation cost and number of parameters while maintaining similar (or slightly better) performance. 对空间信息和深度信息进行去耦合，分别计算depthwise和pointwise的卷积。
  - 假设有一个3×3大小的卷积层，其输入通道为16、输出通道为32。具体为，32个3×3大小的卷积核会遍历16个通道中的每个数据，从而产生16×32=512个特征图谱。进而通过叠加每个输入通道对应的特征图谱后融合得到1个特征图谱。最后可得到所需的32个输出通道。
  - 针对这个例子应用深度可分离卷积，用1个3×3大小的卷积核遍历16通道的数据，得到了16个特征图谱。在融合操作之前，接着用32个1×1大小的卷积核遍历这16个特征图谱，进行相加融合。这个过程使用了16×3×3+16×32×1×1=656个参数，远少于上面的16×32×3×3=4608个参数。
- Decoder: 
  - The encoder features are first bilinearly upsampled by a factor of 4 and then concatenated with the corresponding low-level features from the network backbone that have the same spatial resolution
  - apply another 1 × 1 convolution on the low-level features to reduce the number of channels
  - apply a few 3 × 3 convolutions to refine the features followed by another simple bilinear upsampling by a factor of 4
- Xception
![image.png](deeplabv3plus3.png)
  1. deeper Xception same as *Deformable Convolutional Networks* except that we do not modify the entry flow network structure for fast computation and memory efficiency,
  2. all max pooling operations are replaced by depthwise separable convolution with striding, 
  3. extra batch normalization and ReLU activation are added after each 3 × 3 depthwise convolution


### Auto-DeepLab (Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation)


## Multi-Scale Context Aggregation by Dilated Convolutions (dilated convolution)
- support exponentially expanding receptive fields without losing resolution or coverage
- adapted the VGG-16 network for dense prediction and removed the last two pooling and striding layers.


## RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
![image.png](refinenet1.png)
![image.png](refinenet2.png)
- limitations of DeepLab:
  - First, it needs to perform convolutions on a large number of detailed (high-resolution) feature maps that usually have high-dimensional features, which are computational expensive. Moreover, a large number of high-dimensional and high-resolution feature maps also require huge GPU memory resources, especially in the training stage
  - Second, dilated convolutions introduce a coarse sub-sampling of features, which potentially leads to a loss of important details
- components of **RefineNet Block**:
  - Residual convolution unit: two residual convolution units (RCU) for fine-tuning the pretrained ResNet weights for specific task.
  - Multi-resolution fusion: first applies convolutions for input adaptation, which generate feature maps of the same feature dimension (the smallest one among the inputs), and then upsamples all (smaller) feature maps to the largest resolution of the inputs. Finally, all features maps are fused by summation
  - Chained residual pooling: :a chain of multiple pooling blocks, each consisting of one max-pooling (stride `1`) layer and one convolution layer for capturing background context from a large image region. output feature maps of all pooling blocks are fused together with the input feature map through summation of residual connections
  - Output convolutions: another residual convolution unit (RCU)
- Residual Connection:
  - Short-range: local shortcut connections in one RCU or the residual pooling component
  - Long-range: the connections between RefineNet modules and the ResNet blocks
 

## Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
![image.png](gcn.png)
- the large kernel (and effective receptive field) plays an important role when perform the classification and localization tasks simultaneously
- two design principles:
  - from the localization view, the model structure should be fully convolutional to retain the localization performance and no fully-connected or global pooling layers should be used as these layers will discard the localization information;
  - from the classification view, large kernel size should be adopted in the network architecture to enable densely connections between feature maps and per-pixel classifiers, which enhances the capability to handle different transformations
- pretrained ResNet as feature network and FCN4 as segmentation framework
- **Global Convolutional Network (GCN)**: to generate multi-scale semantic score maps
- score maps of lower resolution will be upsampled with a deconvolution layer, then added up with higher ones to generate new score maps
- **Boundary Refinement (BR)**: models the boundary alignment as a residual structure


## Pyramid Scene Parsing Network (PSPNet)
![image.png](pspnet.png)
- exploit the capability of global context information by different-region-based context aggregation via our pyramid scene parsing network
- start with observation and analysis of representative failure cases when applying FCN methods to scene parsing
- several common issues for complex-scene parsing
  - Mismatched Relationship
  - Confusion Categories
  - Inconspicuous Classes
- pyramid pooling module: fuses features under four different pyramid scales
  - separates the feature map into different sub-regions and forms pooled representation for different locations
  - use `1 × 1` convolution layer after each pyramid level to reduce the dimension of context representation to `1/N` of the original one if the level size of pyramid is `N`
  - directly upsample the low-dimension feature maps to get the same size feature as the original feature map via bilinear interpolation
  - different levels of features are concatenated as the final pyramid pooling global feature
- Deep Supervision
  - final loss and extra classifier


## Deep Extreme Cut: From Extreme Points to Object Segmentation
![image.png](dec.png)
- explores the use of extreme points in an object (left-most, right-most, top, bottom pixels) as input to obtain precise object segmentation for images and videos
- adding an extra channel to the image in the input of a convolutional neural network (CNN), which contains a Gaussian centered in each of the extreme points
- object-centered crop
- balanced loss


## Learning to Segment Every Thing
![image.png](set.png)
- based on `Mask R-CNN`
- Most methods for object instance segmentation require all training examples to be labeled with segmentation masks.
- enables training instance segmentation models on a large set of categories all of which have box annotations, but only a small fraction of which have mask annotations.
- The intuition behind our approach is that once trained, the parameters of the bounding box head encode an embedding of each object category that enables the transfer of visual information for that category to the partially supervised mask head.
- a **partially supervised training paradigm**
  - training on the combination of strong and weak labels
  - redict a category’s mask parameters from its bounding box parameters using a generic, category-agnostic weight transfer function that can be jointly trained as part of the whole model
- a **weight transfer function**
  - trained to predict a category’s instance segmentation parameters as a function of its bounding box detection parameters
  - First, use the COCO dataset to simulate the partially supervised instance segmentation task as a means of establishing quantitative results on a dataset with high-quality annotations and evaluation metrics
  - Second, train a `large-scale` instance segmentation model on 3000 categories using the Visual Genome (VG) dataset



## Multi-Scale Context Intertwining for Semantic Segmentation
![image.png](msci.png)
- merge pairs of feature maps in a bidirectional and recurrent fashion
  - The intertwining is modeled using two chains of long short-term memory (LSTM) units, which repeatedly exchange information between them, in a bidirectional fashion
- subdivide images into super-pixels, and use the spatial relationship between them in order to perform image-adapted context aggregation.
  - subdivide images into super-pixels, and use the spatial relationship between the super-pixel in order to define image-adapted feature connections.
- The intuition here is that semantics and context of adjacent scales are strongly correlated, and hence the descriptive power of the features may be significantly enhanced by such intertwining, leading to more precise semantic labeling.



## SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
![image.png](segnet.png)
- The decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling
  1. it improves boundary delineation, 
  1. it reduces the number of parameters enabling end-to-end training, and 
  1. this form of upsampling can be incorporated into any encoder-decoder architecture such as with only a little modification.
- The encoder network in SegNet is topologically identical to the convolutional layers in VGG16



## MaskLab: Instance Segmentation by Refining Object Detection With Semantic and Direction Features
![image.png](masklab.png)
- based on Faster RCNN
- components:
  - shared feature extractor
  - one extra block is used for the box classifier in Faster-RCNN
  - original conv5 block is shared for both semantic segmentation and direction prediction
  - Semantic segmentation logits and direction prediction logits are computed by another `1 × 1` convolution added after the last feature map on the concatenation of (1) cropped semantic logits from the semantic channel predicted by Faster-RCNN and (2) cropped direction logits after direction pooling
- tasks:
  - box prediction (in particular, refined boxes after the box classifier), 
  - semantic segmentation logits (logits for pixel-wise classification) and 
  - direction prediction logits (logits for predicting each pixel’s direction towards its corresponding instance center)
- Mask refinement: refine the predicted coarse masks by exploiting the hypercolumn features
- Deformable crop and resize: (a) The operation, crop and resize, crops features within a bounding box region and resizes them to a specified size 4 × 4. (b) The 4 × 4 region is then divided into 4 small sub-boxes, and each has size 2 × 2. (c) Another small network is applied to learn the offsets of each sub-box. Then perform crop and resize again w.r.t. to the deformed sub-boxes.


## Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
![image.png](scse.png)
- three variants of SE modules for image segmentation
  1. squeezing spatially and exciting channel-wise (cSE),
  2. squeezing channel-wise and exciting spatially (sSE) and 
  3. concurrent spatial and channel squeeze & excitation (scSE).
- SE blocks only increase overall network complexity by a very small fraction.


## Fully Convolutional Adaptation Networks for Semantic Segmentation
![image.png](fcan.png)
- collecting expert labeled datasets is extremely expensive, rendering synthetic data and generate ground truth automatically is an appealing alternative.
  - high generalization error on real images due to domain shift
- from two perspectives to deal with domain shift
  - appearance-level: adapts source-domain images to appear as if drawn from the “style” in the target domain
  - representation-level: learn domain-invariant representations
- Fully Convolutional Adaptation Networks (FCAN)
  - Appearance Adaptation Networks (AAN): learns a transformation from one domain to the other in the pixel space
    - AAN is to construct an image that captures high-level content in a source image and low-level pixel information of the target domain
  - Representation Adaptation Networks (RAN): optimized in an adversarial learning manner to maximally fool the domain discriminator with the learnt source and target representations
    - a shared Fully Convolutional Networks (FCN) is first employed to produce image representation in each domain, followed by bilinear interpolation to upsample the outputs for pixel-level classification
    - a domain discriminator to distinguish between source and target domain
    - optimizing two losses, i.e., classification loss to measure pixel-level semantics and adversarial loss to maximally fool the domain discriminator with the learnt source and target representations


## CCNet: Criss-Cross Attention for Semantic Segmentation
![image.png](ccnet1.png)
- local receptive fields and short-range contextual information impose a great adverse effect on FCN-based methods due to insufficient contextual information
  - the dilated convolution based methods collect information from a few surrounding pixels and can not generate dense contextual information actually.
  - the pooling based methods aggregate contextual information in a non-adaptive manner and the homogeneous contextual information is adopted by all image pixels, which does not satisfy the requirement the different pixel needs the different contextual dependencies.
![image.png](ccnet2.png)
- Criss-Cross Attention
  - firstly two convolution layers with `1 × 1` filters are applied on **`H`** (shape `C × W × H`) to generate two feature maps **`Q`** and **`K`**, where the shape of **`Q`** and **`K`** is `C' × W × H`. `C'` is the channel number of feature maps, which is less than `C` for dimension reduction
  - generate attention maps **`A`** with shape `(H + W −1) × W × H` via Affinity operation
  - another convolutional layer with `1 × 1` filters is applied on **`H`** to generate **`V`** with shape `C × W × H` for feature adaption.
  - The long-range contextual information is collected by the **Aggregation** operation



# Reference: 
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)
- [计算机视觉 - 常见的图片分类模型【架构演变】 - viredery - 博客园](https://www.cnblogs.com/viredery/p/convolution_neural_networks_for_classification.html)
- [Eason.wxd](https://me.csdn.net/App_12062011)