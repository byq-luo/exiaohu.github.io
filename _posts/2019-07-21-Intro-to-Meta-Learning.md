# Meta-Learning
learning to learn

## 传统机器学习
- 模型
- 优化器

## 元学习
- 模型 -> 低级网络（学习器）
  - 优化方式和传统网络优化方式相同
- 优化器 -> 高级网络（元学习器）
  - 在一组模型训练过程之后，对模型Loss进行反向传播，对元学习器的模型进行优化
  - `meta-loss`：that is indicative of how well the meta-learner is performing its task: training the model.
    - back propagating the meta-loss through the model’s gradients involves computing derivatives of derivative, i.e. second derivatives
    - One possibility is then to compute the loss of the model on some training data, the lower the loss, the better the training was. We can compute a meta-loss at the end or even just combine the losses of the model that we already compute during the training.
  - `meta-optimizer`: to update the weights of the optimizer 
    - coordinate-sharing: design the optimizer for a single parameter of the model and duplicate it for all parameters (i.e. share it’s weights along the input dimension associated to the model parameters). This way the number of parameters of the meta-learner is not a function of the number of parameters of the model. When the meta-learner is a network with a memory like an RNN, we can still allow to have a separate hidden state for each model parameters to keep separate memories of the evolution of each model parameter.
### Reference
[From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)