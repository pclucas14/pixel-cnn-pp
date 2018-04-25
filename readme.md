## PixelCNN++

A Pytorch Implementation of [PixelCNN++.](https://arxiv.org/pdf/1701.05517.pdf)

Main work taken from the [official implementation](https://github.com/openai/pixel-cnn)

Pre-trained models are available [here](https://mega.nz/#F!W7IhST7R!PV7Pbet8Q07GxVLGnmQrZg)

I kept the code structure to facilitate comparison with the official code. 

The code achieves **2.95** BPD on test set, compared to **2.92** BPD on the official tensorflow implementation. 



### Running the code
```
python main.py
```

### Differences with official implementation
1. No data dependant weight initialization 
2. No exponential moving average of past models for test set evalutation

### Contact
For questions / comments / requests, feel free to send me an email.\
Happy generative modeling :)
