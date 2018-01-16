A Pytorch Implementation of PixelCNN++.(https://arxiv.org/pdf/1701.05517.pdf)

Main work taken from the official implementation (https://github.com/openai/pixel-cnn)

Redid in Pytorch cuz why not. I kept the code structure to facilitate comparison with the official code. 

The code achieves 2.95 BPD on test set, compared to 2.92 BPD on the official tensorflow implementation. The main differences are in weight initialization and keeping an exponential moving average of past models for test set evalutation. 
