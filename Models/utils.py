# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:06:39 2020

@author: Rahul K
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle 
import matplotlib.image as mi
from imgaug import augmenters as aug
import cv2
import random

from tensorflow import keras
from tensorflow.keras.layers import Dense, Convolution2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def Name(name):
    return name.split('\\')[-1]
def ImageData(path):
    
    columns = ['Center', 'Left', 'Right', 'Steering','Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    print(data['Center'][0])
    #print(Name(data['Center'][0]))
    #data['Center'] = data['Center'].apply(Name)
    print(f'Total images: {data.shape[0]}')
    
    return data
    
def normalizeData(data):
    Bins = 31
    samples = 300
    
    hist, bins = np.histogram(data['Steering'], Bins)
    print(bins)
    
    norm_bins = (bins[:-1]+bins[1:])*0.5
    print(norm_bins)
    
    plt.bar(norm_bins,hist, width=0.06)
    plt.plot((-1,1),(samples,samples))
    plt.show()
    
    removeIndex = []
    for j in range(Bins):
        bindata = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                bindata.append(i)
        bindata = shuffle(bindata)
        bindata = bindata[samples:]
        removeIndex.extend(bindata)
    print(f'removed Images: {len(removeIndex)}')
    data.drop(data.index[removeIndex], inplace = True)
    print('len of images', len(data))
    
    hist, _ = np.histogram(data['Steering'], Bins)
    plt.bar(norm_bins,hist, width=0.06)
    plt.plot((-1,1),(samples,samples))
    plt.show()
    
    return data

def LoadData(path, data):
    imagePath = []
    Steering_angle = []
    
    for i in range(len(data)):
        indexData = data.iloc[i]
        #print(indexData)
        imagePath.append(indexData[0])
        Steering_angle.append(float(indexData[3]))
        
    imagePath = np.asarray(imagePath)
    Steering_angle = np.asarray(Steering_angle)
        
    return imagePath, Steering_angle



def Augmentation(imgPath, Steering):
    img = mi.imread(imgPath)
    
    ## Shift/Pa
    if np.random.randn() < 0.5:
        pan = aug.Affine(translate_percent={'x':(-0.1,0.1), 'y': (-0.1,0.1)})
        img = pan.augment_image(img)
    
    ## ZOOM
    if np.random.randn() < 0.5:
        zoom = aug.Affine(scale=(1,1.3))
        img = zoom.augment_image(img)
    
    ## Brightness
    if np.random.randn() < 0.5:
        brit = aug.Multiply((0.2,1.2))
        img = brit.augment_image(img)
    
    ## Flip
    if np.random.randn() < 0.5:
        img = cv2.flip(img,1)
        Steering = - Steering
        
    
    return img, Steering

  
def preproxessing(img):
    
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img

        
def batch(imgpath, Steering, batchSize, TrainFlag):
    
    while True:
        
        imgBatch = []
        SteeringBatch = []
        
        for i in range(batchSize):
            index = random.randint(0,len(imgpath)-1)
            
            if TrainFlag:
                img, steering = Augmentation(imgpath[index], Steering[index])
            else:
                img = mi.imread(imgpath[index])
                steering = Steering[index]
            img = preproxessing(img)
            imgBatch.append(img)
            SteeringBatch.append(steering)
        yield(np.asanyarray(imgBatch), np.asarray(SteeringBatch))
        
        
def CarSimulatorModel():
    
    Model = Sequential()
    
    Model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3), activation='elu'))
    Model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    Model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    Model.add(Convolution2D(64,(3,3),activation='elu'))
    Model.add(Convolution2D(64,(3,3),activation='elu'))
    
    Model.add(Flatten())
    Model.add(Dense(100, activation='elu'))
    Model.add(Dense(50, activation='elu'))
    Model.add(Dense(10, activation='elu'))
    Model.add(Dense(1))
    
    Model.compile(Adam(lr=0.0001), loss='mse')
    
    return Model
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        