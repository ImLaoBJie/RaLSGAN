import numpy as np
import zipfile
import os
from time import time
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# update
import read_data_dogimage_v2
from AdamWithWeightnorm import AdamWithWeightnorm

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate, UpSampling2D, Conv2DTranspose, \
    BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras import optimizers
from keras import backend as K
from PIL import Image

if 'tensorflow' == K.backend():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

batch_size = 32
lr = 0.0005
beta1 = 0.5
epochs = 30

nz = 256

# update: add Dropout layers and change optimizers
def convtlayer(input, filter, kernel_size, stride, padding, isdrop=False, drop=0.2):
    x = Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, kernel_initializer='glorot_uniform')(input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-05)(x)
    x = Activation(activation='relu')(x)
    if isdrop:
        x = Dropout(rate=drop)(x)
    return x


def generator():
    input = Input(shape=(nz, ))
    x = Reshape((1, 1, nz))(input)
    x = convtlayer(x, 1024, 4, 1, 'valid')  # as FC layer
    x = convtlayer(x, 512, 4, 2, 'same', isdrop=True)
    x = convtlayer(x, 256, 4, 2, 'same', isdrop=True)
    x = convtlayer(x, 128, 4, 2, 'same', isdrop=True)
    x = convtlayer(x, 64, 4, 2, 'same')
    x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = Activation(activation='tanh')(x)

    gene = Model(input, x, name="generator")
    gene.compile(loss='binary_crossentropy', optimizer=AdamWithWeightnorm(lr=lr, beta_1=beta1))
    gene.summary()

    return gene


def convlayer(input, filter, kernel_size, stride, padding, bn=False, isdrop=False, drop=0.2):
    x = Conv2D(filters=filter, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, kernel_initializer='glorot_uniform')(input)
    if bn:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    if isdrop:
        x = Dropout(rate=drop)(x)
    return x


def discriminator():
    input = Input(shape=(64, 64, 3))
    x = convlayer(input, 32, 4, 2, 'same', isdrop=True)
    x = convlayer(x, 64, 4, 2, 'same', isdrop=True)
    x = convlayer(x, 128, 4, 2, 'same', bn=True, isdrop=True)
    x = convlayer(x, 256, 4, 2, 'same', True)
    x = Conv2D(filters=1, kernel_size=4, strides=1, padding='valid', use_bias=False)(x)  # as FC layer

    disc = Model(input, x, name="discriminator")
    disc.compile(loss='binary_crossentropy', optimizer=AdamWithWeightnorm(lr=lr, beta_1=beta1))
    disc.summary()

    return disc


def train():
    gene = generator()
    disc = discriminator()

    real_img = Input(shape=(64, 64, 3))
    noise_input = Input(shape=(nz,))

    # Build Generator Network
    fake_img = gene(noise_input)
    # Build 2 Discriminator Networks (one from noise input, one from generated samples)
    disc_real = disc(real_img)  # C(x_r)
    disc_fake = disc(fake_img)  # C(x_f)

    # Build Loss
    disc_real_average = K.mean(disc_real, axis=0)
    disc_fake_average = K.mean(disc_fake, axis=0)

    def lossD(y_true, y_pred):
        # epsilon=0.000001
        # return -(K.mean(K.log(K.sigmoid(disc_real_average - disc_fake_average) + epsilon), axis=0) + K.mean(K.log(1 -
        # K.sigmoid(disc_fake_average - disc_real_average) + epsilon), axis=0))
        return K.mean(K.pow(disc_real_average - disc_fake_average - 1, 2), axis=0) + K.mean(
            K.pow(disc_fake_average - disc_real_average + 1, 2), axis=0)

    def lossG(y_true, y_pred):
        # epsilon=0.000001
        # return -(K.mean(K.log(K.sigmoid(disc_fake_average - disc_real_average) + epsilon), axis=0) + K.mean(K.log(1 -
        # K.sigmoid(disc_real_average - disc_fake_average) + epsilon), axis=0))
        return K.mean(K.pow(disc_fake_average - disc_real_average - 1, 2), axis=0) + K.mean(
            K.pow(disc_real_average - disc_fake_average + 1, 2), axis=0)

    # Build Optimizers
    # adamOP = Adam(lr=lr, beta_1=beta1)
    # update: change optimizers
    adamOP = AdamWithWeightnorm(lr=lr, beta_1=beta1)

    # Build trainable generator and discriminator
    disc_train = Model([noise_input, real_img], [disc_real, disc_fake])
    gene.trainable = False
    disc.trainable = True
    disc_train.compile(optimizer=adamOP, loss=[lossD, None])
    disc_train.summary()

    gene_train = Model([noise_input, real_img], [disc_real, disc_fake])
    gene.trainable = True
    disc.trainable = False
    gene_train.compile(optimizer=adamOP, loss=[lossG, None])
    gene_train.summary()

    # Start training
    ideal_target = np.zeros((batch_size, 1), dtype=np.float32)

    # Load data
    images, names = read_data_dogimage_v2.load_data()
    images = images / 255 * 2 - 1

    batch_num = int(len(images[0:]) // (batch_size * 2))

    loss_d = []
    loss_g = []

    for epoch in np.arange(epochs):
        np.random.shuffle(images)

        print('current step: {:d} / {:d}, {:.2f}'.format((epoch + 1), epochs, (epoch + 1) / epochs))
        start_time = time()

        for batch_i in np.arange(0, batch_num):
            batch = images[batch_i * (batch_size * 2): (batch_i + 1) * (batch_size * 2)]

            # The result may be affected by the order or the frequency of training gene or disc per epoch.

            batch_sec = batch[0 * batch_size: 1 * batch_size]
            # noise = np.random.randn(batch_size, nz).astype(np.float32)
            noise = truncnorm.rvs(-1.0, 1.0, size=(batch_size, nz)).astype(np.float32)
            gene.trainable = True
            disc.trainable = False
            loss_g.append(gene_train.train_on_batch([noise, batch_sec], ideal_target))

            batch_sec = batch[1 * batch_size: 2 * batch_size]
            # noise = np.random.randn(batch_size, nz).astype(np.float32)
            noise = truncnorm.rvs(-1.0, 1.0, size=(batch_size, nz)).astype(np.float32)
            gene.trainable = False
            disc.trainable = True
            loss_d.append(disc_train.train_on_batch([noise, batch_sec], ideal_target))

        print('lossG: {}, lossD: {}'.format(loss_g[-1][0], loss_d[-1][0]))
        print('epoch time: {}\n'.format(time() - start_time))

    plt.plot(np.array(loss_g)[:, 0])
    plt.plot(np.array(loss_d)[:, 0])
    plt.legend(['generator', 'discriminator'])
    plt.savefig('loss.png')

    return gene


class ImageGenerator:
    act = 0

    def __init__(self, gene):
        self.gene = gene

    def get_fake_img(self):
        # noise = np.random.randn(1, nz).astype(np.float32)
        noise = truncnorm.rvs(-1.0, 1.0, size=(1, nz)).astype(np.float32)
        img = ((self.gene.predict(noise)[0].reshape((64, 64, 3)) + 1) / 2) * 255
        self.act = (self.act + 1) % 10000
        return Image.fromarray(img.astype('uint8'))


if __name__ == '__main__':
    gene = train()
    I = ImageGenerator(gene)

    z = zipfile.PyZipFile('images.zip', mode='w')
    for k in range(10):
        img = I.get_fake_img()
        f = str(k) + '.png'
        img.save(f, 'PNG')
        z.write(f)
        os.remove(f)
        if k % 1000 == 0:
            print(k)
    z.close()
    print('completed')

