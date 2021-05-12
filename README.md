# VUR-Net
PyTorch model VUR-Net for phase unwrapping

This is a PyTorch realisation of deep convolutional Unet-like network with extra convolutional layers, described in arcticle [1].    

# Changes
I've added following moments to the structure:

1. Replication padding mode in conv3x3 blocks, because experiments have shown that it's important at the edges of phase maps,
otherwise unwrapping quality will be low

# Dataset
Dataset was generated synthetically according to articles [1,2]
So, dataset data was generated using two methods (in equal proportions):

1. Interpolation of squared matrixes (with uniformly distributed elements) of different sizes (2x2 to 15x15) to 256x256 and multiplying by random value, so the magnitude is between 0 and 22 rad
2. Randomly generated Gaussians on 256x256 field with random quantity of functions, means, STD, and multiplying by random value, so the magnitude is between 2 and 20 rad. From experiments with real simple phase images it's clear, that that method makes net more adapted for real-life examples

![example1](https://user-images.githubusercontent.com/73649419/115595429-95d36d00-a2df-11eb-8d83-1a629635a66f.png)
![example2](https://user-images.githubusercontent.com/73649419/115595433-97049a00-a2df-11eb-95d0-73c631d73240.png)

# Model
Model can be shown as following (from original article [1]):

<img src="https://user-images.githubusercontent.com/73649419/116898237-cea80600-ac3e-11eb-9e06-d1c41fd4200a.jpg" data-canonical-src="https://user-images.githubusercontent.com/73649419/116898237-cea80600-ac3e-11eb-9e06-d1c41fd4200a.jpg" width="682" height="864" align="center"/>

# Training info
In original paper authors describe train hyperparameters as follows:

loss: pixelwise MAE

optimizer: Adam 

learning rate: 2e-4 and "divided by 2 when learning stagnates"

Succeed train to zero cost (0.0162) at epoch 500 with Adam 0.0001 very fast

# References
1. Qin, Y., Wan, S., Wan, Y., Weng, J., Liu, W., & Gong, Q. (2020). Direct and accurate phase unwrapping with deep neural network. Applied optics, 59 24, 7258-7267 .
2. K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).
3. Spoorthi, G. E. et al. “PhaseNet 2.0: Phase Unwrapping of Noisy Data Based on Deep Learning Approach.” IEEE Transactions on Image Processing 29 (2020): 4862-4872.
