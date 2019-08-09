# Relativistic average Least Square GAN

This is an implementation of a Relativistic Average GAN with Keras(TensorFlow).

### Reference
* [[paper]](https://arxiv.org/abs/1807.00734)
* [[blog]](https://ajolicoeur.wordpress.com/relativisticgan)
* [[ozanciga/gans-with-pytorch]](https://github.com/ozanciga/gans-with-pytorch)

### Loss function
![loss function](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/lossfunction.png "loss function")

# Quick Start

1. Download dataset from [here](https://www.kaggle.com/c/generative-dog-images/data).
If you want to use CIFAR-100 dataset, you can download `read_data_cifar100.py` and import it from [one of my repository](https://github.com/ImLaoBJie/yolo3sort) to load data.

2. Unzip the dataset to the folder storing `RaLSGAN.py`.

3. Open `RaLSGAN.py` and modify the parameter below£º

```
batch_size = 32
lr = 0.0005  # learning rate
beta1 = 0.5  # optimizer's parameter
epochs = 30

nz = 256  # the dim of noise
```

4. Run `RaLSGAN.py`

# Result

### Loss
![loss](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/loss.png "loss")

### Generated images
| epoch | 1 | 2 | 3 | 4 |
| -------- | -------- | ------------- | -------- |
