#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:03:48 2021

@author: prerna
"""


import cv2
from matplotlib.pyplot import imshow
from tensorflow.keras import Sequential
from tensorflow.keras import Conv2D, Maxpooling2D,UpSampling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

SIZE = 256
img = cv2.imread('/Users/prerna/Downloads/dog.jpeg',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(SIZE,SIZE))
imshow(img)



img_data = []
img_data.append(img_to_array(img))

img_array = np.reshape(img_data,(len(img_data),SIZE,SIZE,3))
img_array = img_array.astype('float32')/255.

model = Sequential()

model.add(Conv2D(32,(3,3),activation= 'relu',padding = 'same',input_shape=(256,256,3)))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(8,(3,3),activation= 'relu',padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(8,(3,3),activation= 'relu',padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))


model.add(Conv2D(8,(3,3),activation= 'relu',padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(8,(3,3),activation= 'relu',padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32,(3,3),activation= 'relu',padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(3,(3,3),activation= 'relu',padding = 'same'))


model.compile(optimizer = 'adam',loss = 'mse',metrics = ['accuracy'])
model.summary()


model.fit(img_array, img_array, epochs = 1000, shuffle = True, verbose = False)

pred = model.predict(img_array)
imshow(pred)
model.fit(img_array, img_array, epochs = 500, shuffle = True)

pred = model.predict(img_array)
imshow(pred[0].reshape(256,256,3))



