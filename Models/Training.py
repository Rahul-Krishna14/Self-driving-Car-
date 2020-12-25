# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:04:29 2020

@author: Rahul K
"""
from utils import *
from sklearn.model_selection import train_test_split

### load path where images are saved
path = r'D:\car simulator\data'

#Calling ImageData function from utils
data = ImageData(path)

## Normalizing the data
data = normalizeData(data)

#Storing the opp path and Steering angles to a array
ImagePath, Steering_angle = LoadData(path, data)

## Spliting the Data into train and Dev set
Xtrain, Xdev, Ytrain, Ydev = train_test_split(ImagePath, Steering_angle, test_size = 0.1, random_state = 47)

print(f'total train{len(Xtrain)}')

## Calling the CNN model created from utils
Model = CarSimulatorModel()

## View the model
Model.summary()

## Fitting the data into the model
Model.fit(batch(Xtrain, Ytrain, 50, 1), steps_per_epoch = 200, epochs = 10, validation_data=batch(Xdev,Ydev,50,0),validation_steps = 200)


#batch(Xtrain, Ytrain, 10, 1) ###test 

## Saving the trained weights into a h5file so that it can be used in application
Model.save(r'D:\car simulator\data\car_simulatorModel.h5')

print("Model Save complete")

## plotting the graph to show performance of the model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Dev'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()