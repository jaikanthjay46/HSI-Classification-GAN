# -*- coding: utf-8 -*-
import cv2
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D,Conv1D,multiply
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D, UpSampling1D
from keras.initializers import RandomNormal
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from skimage.color import label2rgb
import matplotlib.pyplot as plt

import keras.backend as K
K.set_learning_phase(1)

from keras import applications, layers
from numpy.random import seed
from tensorflow import set_random_seed
seed(42)
set_random_seed(42)


#load dataset
import numpy as np
data = np.load('dataset/ds.npy')
gt = np.load('dataset/gt.npy')
H = data.shape[0]
W = data.shape[1]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from patchify import patchify, unpatchify


def define_discriminator(in_shape=(64,64,3), n_classes=17):
	model = Sequential()

	model.add(Conv2D(64, kernel_size=(4,4), strides=2, input_shape=in_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, kernel_size=(4,4), strides=2, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))

	model.add(Conv2D(256, kernel_size=(4,4), strides=2, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))

	model.add(Conv2D(512, kernel_size=(4,4), strides=2, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))

	model.add(Conv2D(64, kernel_size=(4,4), strides=1, padding="same"))

	model.add(Flatten())


	img = Input(shape=in_shape)
	features = model(img)
	validity = Dense(1, activation="sigmoid")(features)
	label = Dense(n_classes, activation="softmax")(features)
	model = Model(img, [validity, label])
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.summary() 
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt,metrics=['accuracy'])
	return model

def define_generator(latent_dim, n_classes=17):
	model = Sequential()

	model.add(Dense(2*2*512, activation="relu", input_dim=latent_dim))
	model.add(Reshape((2,2, 512)))
	model.add(BatchNormalization(momentum=0.8))
	# model.add(UpSampling2D(size=4))

	model.add(Conv2DTranspose(512, kernel_size=(4,4), strides=2, padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(momentum=0.8))

	# model.add(UpSampling2D(size=4))

	model.add(Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(momentum=0.8))

	# model.add(UpSampling2D(size=4))

	model.add(Conv2DTranspose(128, kernel_size=(4,4), strides=2, padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(momentum=0.8))

	# model.add(UpSampling2D(size=4))

	model.add(Conv2DTranspose(64, kernel_size=(4,4), strides=2, padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(momentum=0.8))

	# model.add(UpSampling2D(size=4))

	model.add(Conv2DTranspose(3, kernel_size=(4,4), strides=2, padding='same'))
	model.add(Activation("tanh"))
	model.summary()

	noise = Input(shape=(latent_dim,))
	label = Input(shape=(1,), dtype='int32')
	label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))

	model_input = multiply([noise, label_embedding])
	img = model(model_input)
	model.summary()
	return Model([noise, label], img)






voxel = data.reshape(-1,data.shape[2])
scaler = StandardScaler()
voxel = scaler.fit_transform(voxel)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(voxel)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc'+str(x) for x in range(3)])
fullDataX = principalComponents.reshape(H,W,3)
fullDataY = gt


paddedDatax = cv2.copyMakeBorder( fullDataX, 32, 31, 32, 31, cv2.BORDER_REPLICATE)


X_train = patchify(paddedDatax, (64, 64, 3), step=1).reshape(-1,64,64,3)
y_train = fullDataY.reshape(-1,)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.50)

def generate_real_test_samples(n_samples):
	images, labels =  X_test, y_test
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = ones((n_samples, 1))
	return [X, labels], y

def generate_real_samples(n_samples):
	images, labels =  X_train, y_train
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = ones((n_samples, 1))
	return [X, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes=17):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict([z_input, labels_input])
	y = zeros((n_samples, 1))
	return [images, labels_input], y

def summarize_performance(step, g_model, latent_dim):
	filename = 'model_%04d.h5' % (step+1)
	g_model.save_weights(filename)
	!cp -f *.h5 /content/drive/My\ Drive/models	

def train(g_model, d_model, gan_model, latent_dim, n_epochs=600, n_batch=4096):
	import warnings
	warnings.filterwarnings("ignore")
	bat_per_epo = int(10000 / n_batch)
	n_steps = bat_per_epo * n_epochs
	half_batch = int(n_batch / 2)
	prev_acc1 = 0.20
	i = 0
	while True:
		[X_real, labels_real], y_real = generate_real_samples( half_batch)
		
		d_loss_real = d_model.train_on_batch(X_real, [y_real, labels_real])
		[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		d_loss_fake = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
		[z_input, z_labels] = generate_latent_points(latent_dim, half_batch)
		y_gan = ones((half_batch, 1))
		g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
		op_acc = d_loss[4]
		trainHist.append([d_loss,g_loss])
		print ("Training Metrics: %d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

		[X_treal, labels_treal], y_treal = generate_real_test_samples(half_batch)
		d_loss_real = d_model.test_on_batch(X_treal, [y_treal, labels_treal])
	
		[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		d_loss_fake = d_model.test_on_batch(X_fake, [y_fake, labels_fake])
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
		

		[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
		y_gan = ones((n_batch, 1))
		g_loss = gan_model.test_on_batch([z_input, z_labels], [y_gan, z_labels])
		validationHist.append([d_loss,g_loss])
		print ("Validation Metrics: %d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
		
		if (op_acc > 0.95 and d_loss[4] > 0.95) or i > 6000:
			prev_acc1 = op_acc
			prev_acc2 = d_loss[4]
			summarize_performance(i, d_model, latent_dim)
			break
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, d_model, latent_dim)
		i += 1

latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)

discriminator.trainable = False
gan_output = discriminator(generator.output)
gan_model = Model(generator.input, gan_output)
opt = Adam(lr=0.002, beta_1=0.5)
gan_model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
  

train(generator, discriminator, gan_model, latent_dim)




