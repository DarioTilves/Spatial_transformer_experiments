# Spatial Transformer Experiments

Pytorch implementation of [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) using as baseline the [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html). The convolutional layers were modified with [CoordConv](https://arxiv.org/abs/1807.03247) to see if it can improve the baseline. Finally, a Spatial transformer extension was explored, [A Reinforcement Learning Approach for Sequential Spatial Transformer Networks](https://arxiv.org/abs/2106.14295).

## Repository content

```
├── README.md
├── requirements.txt
├── images_resources/ # Folder to store the images shown in the README.md
├── logs/ # Folder used to save the experiments
└── python/  
    ├── config/  
    │   └──config.yml # Configuration file to change training parameters
    └── core_code 
        ├── metrics/
        │   └── metric.py
        ├── models/
        │   ├── coord_conv.py # CoordConv implementation
        │   ├── losses.py # Losses used in the reinforcement learning transformer
        │   ├── reinforced_spatial_transformer.py # Implementation of the paper A Reinforcement Learning Approach for Sequential Spatial Transformer Networks
        │   ├── sequential_spatial_transformer.py # Middle ground implementation between the baseline and the Reinforcement learning approach
        │   └── spatial_transformer.py # Baseline model
        ├── main.py # Script to train and test the models
        └─  utils.py 

```

## Run models

Move to `requirements.txt` folder level and to install all the libraries required run:

```sh
pip install -r requirements.txt
```

 Mind you it will install torch and torchvision libraries with cuda 11.3 support. To run the training script:

```sh
python python/core_code/main.py
```

In order to select the model trained and modify training and visualization parameters is necessary to modify `config.yml` insite config folder:

```
model_name:             Spatial_transformer_conv_conv # Name used to save the model in logs folder
model_selection:
  Spatial_transformer:  0     # 0 Spatial transformer, 1 sequential transformer, 2 reinforced transformer
  iterations:           2     # Minimum value of 2 for reinforced transformer, minimum value of 1 for sequential transformer and it does not affect spatial transformer
  loss_gamma:           0.98  # Penalty value used by the reinforcement learning algorithm
conv_coord_selection:         # Any model can use convolutional layers or ConvCoord layers, set to False to use ConvCoord layers
  conv_localisation:    True  # Modify conv layers before affine transformation, set to False to use ConvCoord layers
  conv_classification:  True  # Modify conv layers in the classification network, set to False to use ConvCoord layers
dataloader_parameters:        # These modifications affect both train and test dataloaders
  batch_size:           64  
  shuffle:              True
  num_workers:          0
model_hyperparameters:
  epoch_end:            5     # Number of epochs the model is trained 
  learning_rate:        0.01  # Default value of 0.01 for spatial and sequential transformer and 0.001 for reinforced transformer
visualizations:
  print_metrics:        True  # Show metrics on the terminal at the end of each epoch
  save_metrics:         True  # Save metrics in the log folder
  save_cf_matrix:       True  # Save last confusion matrix in the log folder
  show_cf_matrix:       False # Plot last confusion matrix
  save_stn:             True  # Save last affine transformation in the log folder
  show_stn:             False # Plot last affine transformation
  ```

## Experiments

First, it was explored the advancement the [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) brought to the table. Being able to transform the input image to enhance the performance of the classification network makes the identification problem not only easier for the neural network but also for humans to interpret. I adapted the [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html) to my own code standard.

The MNIST dataset is going to be used in all the experiments.

[CoordConv](https://arxiv.org/abs/1807.03247) improves network mapping capabilities as a result of specifying each pixel coordinate inside the convolution. Howewer, this can reduce the translation invariability convolutional layers have, therefore it could worsen performance in classification task. So to test if the affine transformations were able to "normalize" each image class [CoordConv](https://arxiv.org/abs/1807.03247) mapping capabilities were used. Uber original [implementation in Tensorflow](https://github.com/uber-research/coordconv) was used as base.

When I was reading [A Reinforcement Learning Approach for Sequential Spatial Transformer Networks](https://arxiv.org/abs/2106.14295) it occured to me I could add a LSTM layer in the spatial transformer localisation network and iterate over transformed images as if they were a sequence. Maybe each step would have worse transformations than the spatial transformer but adding those little steps can lead to better performance.

Finally, I have tried to implement [A Reinforcement Learning Approach for Sequential Spatial Transformer Networks](https://arxiv.org/abs/2106.14295) because I was interested in the idea of reinforcement learning applied to spatial transformers networks. In this case, Azimi et al. modified the localisation network to act as a policy network. The paper proposes that the affine transformations are predefined so the problem is transformed to a sequence of simple and discrete transformations using reinforcement learning to solve this decision problem. 

![](https://i.loli.net/2018/07/16/5b4c8d88a70a2.png)

As seen in the image, the spatial transformer localisation network plus a LSTM and softmax layers form the Policy network. The Policy network output is the action ,i.e., the affine transformation. The state is composed of the current image and the previous action. The reward function is the substraction between the previous transformed image classification loss and the current image classification loss. As there is no code available I have implemented these concepts only by the papers' descriptions. Mind you that this is my first time exploring reinforcement learning neural nets (this was the reason I have tried to do it, to learn how reinforcement learning works), so any points and corrections are welcome.

### Results

First, the differents models are going to be compared when trained the same number of epochs (5), same batch size (64) and same learning rate (0,01). Varying the application of CoordConv and the number of images iterated over the LSTM.

Spatial Transformer Network = STN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; CoordConv = CC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sequential STN = SSTN
Reinforcement learning SSTN = RLSSTN &nbsp; Not applicable = NA &nbsp; Learning rate = LR

| Model     |LSTM iterations| Accuracy | Precision | Sensitivity | Specifity | F1 Score|
| :---:     |:---:          | :---:    | :---:     | :---:       | :---:     | :---:   |
| STN       | NA            | 99,67%   | 98,35%    | 98,37%      | 99,82%    | 98,35%  |
| STN CC    | NA            | 99,56%   | 97,26%    | 97,80%      | 99,75%    | 97,81%  |
| SSTN      | 2             | 99,34%   | 96,73%    | 96,69%      | 99,64%    | 96,70%  |
| SSTN CC   | 2             | 99,25%   | 96,27%    | 96,23%      | 99,58%    | 96,24%  |
| SSTN      | 5             | 99,37%   | 96,87%    | 96,85%      | 99,65%    | 96,85%  |
| SSTN CC   | 5             | 99,21%   | 96,05%    | 96,03%      | 99,56%    | 96,03%  |
| RLSSTN    | 2             | 97,40%   | 87,11%    | 86,93%      | 98,55%    | 86,92%  |
| RLSSTN CC | 2             | 96,91%   | 85,34%    | 84,45%      | 98,28%    | 84,46%  |
| RLSSTN    | 5             | 93,57%   | 69,06%    | 67,56%      | 96,42%    | 67,52%  |
| RLSSTN CC | 5             | 92,91%   | 67,00%    | 64,24%      | 96,06%    | 64,10%  |

As can be seen from the table above, CoordConv is worsening a little the models performance. It was commented before that is not the best algorithm to apply to classification tasks due to reducing convolution translation invariance. 

On the other hand, both SSTN and RLSSTN also performed worse than the baseline. The iteration of the images over the LSTM is not a better strategy, which for the simple dataset MNIST can be true. If one transformation is enough to almost achieve a perfect result it is logical to think several would hindrance the performance. SSTN should be tested with a more challenging dataset to really see the performance impact. Furthermore, when we add more LSTM iterations SSTN performance is the same (within the error margin) which means it has learned not to impact the performance unlike RLSSTN. 

However, RLSSTN results are worse due to poor decision on the affine transformations. As explained before, RLSSTN affine transformations are predefined by hand, in this case: identity matrix, 2 translations, 2 rotations, 2 scaling and 2 shear operations. Moreover, these operations are not complementary so they cannot be undone, as SSTN probably learn to do, so stacking more LSTM iterations on RLSSTN would only make the model worse. 

Secondly, CoordConv performance is going to be analysed while applied to different parts of the STN.

CoordConv implemented only on localisation (network first part) = CCL
CoordConv implemented only on classification (network last part) = CCF

| Model     |LSTM iterations| Accuracy | Precision | Sensitivity | Specifity | F1 Score|
| :---:     |:---:          | :---:    | :---:     | :---:       | :---:     | :---:   |
| STN       | NA            | 99,67%   | 98,35%    | 98,37%      | 99,82%    | 98,35%  |
| STN CC    | NA            | 99,56%   | 97,26%    | 97,80%      | 99,75%    | 97,81%  |
| STN CCL   | NA            | 99,66%   | 98,28%    | 98,29%      | 99,81%    | 98,28%  |
| STN CCF   | NA            | 99,32%   | 96,62%    | 96,57%      | 99,62%    | 96,57%  |
| SSTN      | 2             | 99,34%   | 96,73%    | 96,69%      | 99,64%    | 96,70%  |
| SSTN CC   | 2             | 99,25%   | 96,27%    | 96,23%      | 99,58%    | 96,24%  |
| SSTN CCL  | 2             | 99,30%   | 96,45%    | 96,46%      | 99,61%    | 96,45%  |
| SSTN CCF  | 2             | 98,61%   | 93,31%    | 92,98%      | 99,22%    | 92,98%  |

With these results it can be concluded that CoordConv is worsening the classification task. When it is applied during the feature extraction it does not have an impact in the performance. SSTN CCF performance could be worse because the transformations are more different between samples so CoordConv mapping is hurting it more.

Thirdly, we are going to see how RLSSTN performance is affected by the learning rate, number of epochs trained and LSTM iterations.

|Model |Epochs|LR   |LSTM iterations| Accuracy | Precision | Sensitivity | Specifity | F1 Score|
|:---: |:---: |:---:|:---:          | :---:    | :---:     | :---:       | :---:     | :---:   |
|RLSSTN| 5    |0.01 | 2             | 97,40%   | 87,11%    | 86,93%      | 98,55%    | 86,92%  |
|RLSSTN| 5    |0.001| 2             | 97,90%   | 89,70%    | 89,43%      | 98,83%    | 89,47%  |
|RLSSTN| 5    |0.01 | 5             | 93,57%   | 69,06%    | 67,56%      | 96,42%    | 67,52%  |
|RLSSTN| 5    |0.001| 5             | 94,58%   | 73,13%    | 72,76%      | 96,99%    | 72,53%  |
|RLSSTN| 10   |0.01 | 2             | 97,40%   | 87,51%    | 86,95%      | 98,56%    | 87,00%  |
|RLSSTN| 10   |0.001| 2             | 98,12%   | 90,75%    | 90,53%      | 98,98%    | 90,57%  |
|RLSSTN| 10   |0.01 | 5             | 94,07%   | 72,11%    | 70,06%      | 96,71%    | 70,37%  |
|RLSSTN| 10   |0.001| 5             | 95,11%   | 75,58%    | 75,38%      | 97,28%    | 75,33%  |

The table above shows that with a smaller learning rate the RL algorithm is clearly benefited. [Azimi et al.](https://arxiv.org/abs/2106.14295) stated the perfert learning rate at 0.0001 and the optimal number of LSTM iterations at 20.

Lastly, there are 5 pictures depicting how does the different models modify MNIST images. All models were trained with same batch size (64), number of epochs (5) and learning rate (0,01).

|STN|SSTN 2 LSTM iterations|SSTN 5 LSTM iterations|LRSSTN 2 LSTM iterations|LRSSTN 5 LSTM iterations|
|:---:|:---:|:---:|:---:|:---:|
|![](https://i.loli.net/2018/07/16/5b4c7db11abf9.png)|![](https://i.loli.net/2018/07/16/5b4c7dbd03169.png)|![](https://i.loli.net/2018/07/16/5b4c8d88a70a2.png)|![](https://i.loli.net/2018/07/16/5b4c7dbd03169.png)|![](https://i.loli.net/2018/07/16/5b4c8d88a70a2.png)|

On the left we can see the SPT transformation is minimal there are only some translation and shearing. Bpth SSTN rotate the numbers 90º and the 5 LSTM iteration adjusts the numbers better to the available space making them bigger. Finally, RLSSTN applies all affine transformations at once. Comparing the two RLSSTN it is fair to say that more transformations achieve worse results according to, as we say before, not being able to reverse the transformations.

## References

Spatial Transformer Networks:
* [Spatial Transformer Networks paper](https://arxiv.org/abs/1506.02025)
* [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
* [SPT A Self-Contained Introduction - Towardsdatascience article](https://towardsdatascience.com/spatial-transformer-networks-b743c0d112be)

CoordConv:
* [Uber paper](https://arxiv.org/abs/1807.03247)
* [Uber implementation](https://github.com/uber-research/coordconv)
* [Tutorial: An introduction to Uber’s new CoordConv - Medium article](https://medium.com/@Cambridge_Spark/coordconv-layer-deep-learning-e02d728c2311)
* [Autopsy Of A Deep Learning Paper - Piekniewski's blog article](https://blog.piekniewski.info/2018/07/14/autopsy-dl-paper/)

Reinforcement Learning:
* [Reinforcement learning - Wikipedia article](https://en.wikipedia.org/wiki/Reinforcement_learning)
* [Reinforcement Learning Sequential Spatial Transformer paper](https://arxiv.org/abs/2106.14295)
* [Distribution implemented](https://pytorch.org/docs/stable/distributions.html#)
* [Understanding Actor Critic Methods and A2C - Towardsdatascience article](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
* [Affine transformations visualizations video - Youtube Leios Labs ](https://www.youtube.com/watch?v=E3Phj6J287o)

Metrics:
* [Sensitivity and specificity - Wikipedia article](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* [Method used to accumulate confusion matrix values](https://stackoverflow.com/questions/52636288/fastest-way-to-build-numpy-array-from-sum-of-coordinates)