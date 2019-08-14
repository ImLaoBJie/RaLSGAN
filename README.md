# Relativistic average Least Square GAN

This is an implementation of a Relativistic Average GAN with Keras(TensorFlow).

### Reference
* [[paper]](https://arxiv.org/abs/1807.00734)
* [[blog]](https://ajolicoeur.wordpress.com/relativisticgan)
* [[ozanciga/gans-with-pytorch]](https://github.com/ozanciga/gans-with-pytorch)

### Loss function
![loss function](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/lossfunction.png "loss function")

# Update (version 2)

1.Add a customized Adam optimizer using Weight Normalization.

* [[paper]](https://arxiv.org/pdf/1704.03971.pdf)
* [[krasserm/weightnorm]](https://github.com/krasserm/weightnorm/tree/master/keras_2

2. Add Dropout layers.

3. Discriminator and generator are compiled before training.

4. Improve the methods of images pre-processing.

### Generated images

* 1000 epochs:

![v2_1](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/v2_1.png "v2_1") ![v2_2](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/v2_2.png "v2_2") ![v2_3](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/v2_3.png "v2_3") ![v2_4](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/v2_4.png "v2_4")

# Quick Start

1. Download dataset from [here](https://www.kaggle.com/c/generative-dog-images/data).
If you want to use CIFAR-100 dataset, you can download `read_data_cifar100.py` and import it from [one of my repository](https://github.com/ImLaoBJie/yolo3sort) to load data.

2. Unzip the dataset to the folder storing `RaLSGAN.py`.

3. Open `RaLSGAN.py` and modify the parameter below:

```
batch_size = 32
lr = 0.0005  # learning rate
beta1 = 0.5  # optimizer's parameter
epochs = 30

nz = 256  # the dim of noise
```

4. Run `RaLSGAN.py`.

# Result

### Loss
![loss](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/loss.png "loss")

### Generated images

* 10 epochs:

![5](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/5.png "5") ![6](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/6.png "6") ![7](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/7.png "7") ![8](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/8.png "8")
* 1000epochs:

![1](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/1.png "1") ![2](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/2.png "2") ![3](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/3.png "3") ![4](https://raw.githubusercontent.com/ImLaoBJie/RaLSGAN/master/img/4.png "4")
