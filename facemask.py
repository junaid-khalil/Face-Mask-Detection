# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 17:51:54 2023

@author: Toshiba
"""



#importing required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

with_mask_files=os.listdir('data/with_mask')
#listdir will create a list which contains all the file names of withmaskfolder 
# print("no of with mask images is",len(with_mask_files))
# print(with_mask_files[0:5])#print first five file names
# print(with_mask_files[-5:]) #print last five file names

# #without mask files raeding
without_mask_files=os.listdir('data/without_mask')
# print("no of without mask images is",len(without_mask_files))
# print(without_mask_files[0:5])#print first five file names
# print(without_mask_files[-5:]) #print last five file names

#creating labels for with mask and without mask files
with_mask_label=[1]*3725#create a list of 1 of len withmask
without_mask_label=[0]*3828# create a list of 0 of length without mask
# print(with_mask_label[0:5])
# print(without_mask_label[0:5])
#creating a combine list of 0 and 1
labels=with_mask_label+without_mask_label
# print(len(labels))

# #displaying the image with mask
# img=mpimg.imread('data/with_mask/with_mask_1.jpg')
# imgplot=plt.imshow(img)
# plt.show()
# #displaying the image without mask
# img=mpimg.imread('data/without_mask/without_mask_2925.jpg')
# imgplot=plt.imshow(img)
# plt.show()

#converting images into nupy arrays
with_mask_path='data/with_mask/'
data=[]
for img_file in with_mask_files:
    image=Image.open(with_mask_path+img_file)
    image=image.resize((128,128))
    image=image.convert('RGB')
    image=np.array(image)
    data.append(image)
without_mask_path='data/without_mask/'
for img_file in without_mask_files:
    image=Image.open(without_mask_path+img_file)
    image=image.resize((128,128))
    image=image.convert('RGB')
    image=np.array(image)
    data.append(image)
# print(len(data))
# print(data[0])
# print(type(data[0]))
# print(data[0].shape)

#converting image list(data) and label list to numpy arrays
X=np.array(data)
Y=np.array(labels)

# #train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)

#scaling the data
X_train_scaled=X_train/255
X_test_scaled=X_test/255
# print(X_train_scaled[0])

#building convolutional neural network model
import tensorflow as tf
from tensorflow import keras

num_of_classes=2#mask  or not
model=keras.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128 , activation='relu'))
model.add(keras.layers.Dropout(0.5))          

model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.5))    

model.add(keras.layers.Dense(num_of_classes,activation='sigmoid'))
          
#compile the neural network
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

#traing the neural network
history=model.fit(X_train_scaled,Y_train,validation_split=0.1,epochs=5)   

#model evealuation
loss,accuracy=model.evaluate(X_test_scaled,Y_test)
print('Test accuracy',accuracy)

#plotting the evaluation

h=history
#plot the loss value
plt.plot(h.history['loss'],label='train loss')
plt.plot(h.history['val_loss'],label='validation loss')
plt.legend()
plt.show()
#plot the accuracy value
plt.plot(h.history['acc'],label='train accuracy')
plt.plot(h.history['val_acc'],label='validation accuracy')
plt.legend()
plt.show()
      
#prediction
input_image_path=input(" paste path of the image to predict")
input_image=cv2.imread(input_image_path)
plt.imshow(input_image)
input_image_resized=cv2.resize(input_image,(128,128))
input_image_scaled=input_image_resized/255
input_image_reshaped=np.reshape(input_image_scaled,[1,128,128,3])
input_prediction=model.predict(input_image_reshaped)
print(input_prediction)
input_prediction_label=np.argmax(input_prediction)
print(input_prediction_label)

if input_prediction_label==1:
    print("the person is wearing the mask")
else:
    print("person is not wearing mask")
    

















