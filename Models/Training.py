# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:04:29 2020

@author: Rahul K
"""
from utils import *
from sklearn.model_selection import train_test_split

### loading IMG path
path = r'D:\car simulator\data'

# image fun from util 
data = ImageData(path)

# Normalizing
data = normalizeData(data)

# loading IMG and steering angles 
ImagePath, Steering_angle = LoadData(path, data)

## Splitting data
Xtrain, Xdev, Ytrain, Ydev = train_test_split(ImagePath, Steering_angle, test_size = 0.1, random_state = 47)


## Calling our CNN model built from util
Model = CarSimulatorModel()

## Model summary
Model.summary()

## Training model
Model.fit(batch(Xtrain, Ytrain, 50, 1), steps_per_epoch = 200, epochs = 10, validation_data=batch(Xdev,Ydev,50,0),validation_steps = 200)


#batch(Xtrain, Ytrain, 10, 1) ###test 

## Saving model weights as h5 file so it can be used in other applications
Model.save(r'D:\car simulator\data\car_simulatorModel.h5')

print("Model Save complete")

## plotting the graph to show performance of the model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Dev'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
