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

def define_discriminator(in_shape=(10,1), n_classes=17):
	model = Sequential()

	model.add(Conv1D(256, kernel_size=4, strides=1, input_shape=in_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv1D(512, kernel_size=4, strides=1, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))

	model.add(Conv1D(128, kernel_size=4, strides=1, padding="same"))

	model.add(Flatten())
	model.summary()

	img = Input(shape=in_shape)
	features = model(img)
	validity = Dense(1, activation="sigmoid")(features)
	label = Dense(n_classes, activation="softmax")(features)
	model = Model(img, [validity, label])
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.summary()
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
	return model






voxel = data.reshape(-1,data.shape[2])
scaler = StandardScaler()
voxel = scaler.fit_transform(voxel)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(voxel)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc'+str(x) for x in range(10)])
fullDataX = principalDf
fullDataY = pd.DataFrame(data=gt)





#testing phase
GAN = define_discriminator()
GAN.load_weights("models/model_0698.h5")

label_image = GAN.predict( fullDataX.to_numpy().reshape(145*145,10,1) )
label_image = np.array(label_image[1])
y = []
for t in label_image:
  y.append(np.argmax(t))
label_image = np.array(y).reshape(145,145)
image_label_overlay = label2rgb(label_image)
fig, (ax1, ax2, ax3)  = plt.subplots(3,figsize=(10, 10))

ax1.imshow(image_label_overlay)
image_label_overlay = label2rgb(fullDataY.to_numpy().reshape(145,145))
ax2.imshow(image_label_overlay)
image_label_overlay = label2rgb(clf.predict(fullDataX.to_numpy()).reshape(145,145)  )
ax3.imshow(image_label_overlay)



