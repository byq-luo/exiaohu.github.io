---
layout: mypost
title: 图神经网络及其在交通预测方面的应用
categories: [Traffic prediction, Graph Neural Networks]
---

# 交通预测日志

## Adaptive Supports Learning

### step1: node embedding

inputs: $X \in \mathbb{R}^{\zeta \times T \times N \times F}$,

outputs: $E_N \in \mathbb{R}^{N \times D_N}$.

Minimize:
$$
\|X - \hat{X}\| = \|X - X \times_2 E_N \times_2 E_N^T\|,
$$
where $E_N \in \mathbb{R}^{N \times D_N}$。

### step2: Supports Learning

inputs: $X \in \mathbb{R}^{B \times T \times N \times F}$,

ouputs: $L \in \mathbb{R}^{N \times N \times D_E}$.

Edge representation of Node $i$ and Node $j$ is acquired by:
$$
\begin{align*}
L'_{ij} &= MLP([E_N(i);E_N(j)]), \\
L &= norm_1(relu(L')),
\end{align*}
$$
where $norm_1(\cdot)$ is L1 normalization. The input dimension of $MLP$ is $2D_N$, and its output dimension is $D_E$.



## Dynamic Bias Learning

### step1: inputs partition.

inputs: $X \in \mathbb{R}^{B \times T \times N \times F}$,

outputs: $X_s, X_d \in \mathbb{R}^{B \times T \times N \times F}$.
$$
\begin{align*}
X_s &= X \times_2 E_N \times_2 E_N^T, \\
X_d &= X - X_s. 
\end{align*}
$$

### step2: Bias Learning

inputs:  $X_s, X_d \in \mathbb{R}^{B \times T \times N \times F}$,

outputs: $\mathbf{B} \in \mathbb{R}^{B \times N \times N \times D_E}$
$$
\begin{align*}
X &= [X_s;X_d], \\
\mathbf{B}'_{i,j} &= MLP([X_i, X_j]), \\
\mathbf{B} &= norm_1(relu(\mathbf{B}')),
\end{align*}
$$
where the input dimension of $MLP$ is $4F$, its output dimension is $D_E$. At last,
$$
L_d = L + \alpha \mathbf{B},
$$
where $\alpha$ is a hyper-parameter.

