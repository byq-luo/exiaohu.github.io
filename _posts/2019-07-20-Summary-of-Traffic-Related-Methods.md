---
layout: mypost
title: 交通相关的方法整理
categories: [Traffic prediction]
---


## Traffic Flow Prediction With Big Data: A Deep Learning Approach
- model: Stacked Autoencoder (SAE) + FC
- problem: predicting traffic flow on freeways
- input `r` 个 `m` 维的向量，`m`是路段数，`r`是历史时间片数目
- output `m`向量
- 单独预测每个路段的流量，没有画格子也没有建交通图


## ZEST: a Hybrid Model on Predicting Passenger Demand for Chauffeured Car Service
- model
  - temporal predictor: model the influences of local factors
    - 基于规则的方法
  - spatial predictor: model the influences of spatial factors
    - 把临近网格的数据输入人工神经网络(ANN)，非CNN
  - ensemble predictor: combine the results of former two predictors
    - Gradient Boosting Decision Trees
  - Zero-Grid predictor: predict zero demand areas specifically since any cruising within these areas costs extra waste on energy and time of driver、
    - a Gradient Boosting Classifier is trained to predict the possibility of Zero Grid


## DNN-Based Prediction Model for Spatio-Temporal Data
- spatio-temporal
  - spatial: near and distant: **convolution neural network**
  - temporal: **closeness, period and trend**: different timestamps are selected and concatenated together to model closeness, period and trend, respectively.
- global
  - day of the week, weekday or weekend, etc.


## CNN + LSTM/GRU
- 先用 CNN 做 spatial embedding, 然后用 LSTM/GRU 进行预测
  - yu\_spatiotemporal\_2017
  - **yao\_deep\_2018**
- 使用 ConvLSTM/ConvGRU，同时对时空特征进行建模
  - zhou\_predicting\_2018
    - encoder-decoder with attention
  - **zonoozi\_periodic-crn:\_2018**



## LC-RNN: A Deep Learning Model for Traffic Speed Prediction