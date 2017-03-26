
# coding: utf-8

# In[1]:

# Do the import
import numpy as np
import csv
import cv2
import tensorflow as tf
import keras
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from sklearn.utils import shuffle


# In[5]:

#Read the CSV file
lines_of_csv = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines_of_csv.append(line)

#Removing the 0th line as it has names of columns
del lines_of_csv[0]
        
print (lines_of_csv[0])       
print (lines_of_csv[28])
print (len(lines_of_csv))

shuffle(lines_of_csv)

train_data, validation_data = train_test_split(lines_of_csv, test_size=0.4)
print ("CSV file read and the data is split")
print (len(train_data))
print (len(validation_data))


# In[10]:

# Do the data generation

def generate_data (samples, training, batch_size = 32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            #create the empty list of images and angles
            images = []
            angles = []

            for batch_sample in batch_samples:
                center_name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                
                #center_angle
                center_angle = float(batch_sample[3])
                
                #center_image = image_process(center_image)                
                images.append(center_image)
                angles.append(center_angle)
                
                #Flip the data for training only
                if training == True:
                    center_image_flipped = cv2.flip(center_image, 1)
                    #center_image_flipped = np.fliplr(center_image)
                    center_angle_flipped = center_angle * -1.0
                    images.append(center_image_flipped)
                    angles.append(center_angle_flipped)
                    
                    left_name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(left_name)
                
                    right_name = '../data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(right_name)
                
                    #left, right images
                    correction = 0.25
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction
             
                    images.append(left_image)
                    angles.append(left_angle)
                
                    images.append(right_image)
                    angles.append(right_angle)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[11]:

#Do the image processing

def image_process (image):
    #Normalize the image
    
    #Crop the image
    
    #display the images
    return image
    


# In[12]:

#Build the model

training = True
train_generator = generate_data(train_data, training, batch_size=32)
training = False
validation_generator = generate_data(validation_data, training, batch_size=32)


model = Sequential()

# Do the image normalization
model.add(Lambda(lambda x: x/127.5 - 1.,  input_shape=(160, 320, 3)))
# Crop the bonet and thse sky/trees
model.add(Cropping2D(cropping=((70,25), (0,0))))
#Implement the Nvidia E2E Learning Network
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(64,3,3, activation="elu"))
model.add(Convolution2D(64,3,3, activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50)) 
model.add(ELU())
model.add(Dense(10))
model.add(ELU())

#9. Output
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
#model.fit (X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 7)

model.fit_generator(train_generator, samples_per_epoch= len(train_data)*4, validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=3)

model.save('model.h5')


# In[ ]:

#Show the histogram of the steering angle
groups_of_sa = 100
sa_per_group = int (len(measurements)/groups_of_sa)
#sa_per_group += 1
print (sa_per_group)

hist, bins = np.histogram (measurements, sa_per_group)
bins = bins[:-1]

print (hist)
print (bins)

plt.bar ( bins, hist, align = 'center', width = 0.1)
plt.ylabel('Steering angle per group')
plt.xlabel('Total groups ')
plt.show()

