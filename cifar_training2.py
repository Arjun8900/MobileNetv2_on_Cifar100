import keras
from keras.utils.np_utils import to_categorical
import numpy as np
import sys
from keras.datasets import cifar100
from keras.applications.mobilenetv2 import MobileNetV2
import tensorflow
import pickle
import os
import cv2
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Conv2D, Reshape, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Model

num_classes = 100
size = 32
batch = 128

# Shuffling the data


# Pickle File load
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
x_train,y_train = unison_shuffled_copies(x_train,y_train)




#count1 = x_train.shape[0]
#count2 = x_test.shape[0]

#print(count1, ' ' ,count2)
print(x_train.shape)

base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Reshape((1,1,1280))(x)
x = Dropout(0.3, name='Dropout')(x)
x = Conv2D(num_classes,(1,1),padding='same')(x)
x = Activation('softmax', name='softmax')(x)
output = Reshape((num_classes,) )(x)

model = Model(inputs = base_model.input, outputs = output)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=300,validation_split=0.05,  shuffle=True, batch_size=batch)
#hist = model.fit(x_train, y_train, train_generator, steps_per_epoch=count1//batch, validation_steps = count2//batch, epochs = 300)
model.save('mv2_cifar100_2.model')




