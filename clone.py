import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
import cv2
from sklearn.utils import shuffle
BATCH_SIZE=32
IMAGES_PER_SAMPLE=4

filepath='../CardND-Behavioral-Cloning-P3/driving_log.csv'
filepath2='data/driving_log.csv'
lines=[]
with open(filepath2) as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append([line,'data/'])	
print('headers',lines[0])
del(lines[0])
headers2=len(lines)
filepath2='data_recover/driving_log.csv'
with open(filepath2) as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append([line,'data_recover/'])	
		#print('line',line[0])
#print('lines',len(lines))

print('total samples in folder2:',len(lines))
#print('headers',lines[0])
del(lines[headers2])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=BATCH_SIZE, steps_per_sample=4):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size//steps_per_sample):
            batch_samples = samples[offset:offset+batch_size//steps_per_sample]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename=batch_sample[0][0].split('/')[-1]
                name=batch_sample[1]+'/IMG/'+filename
                img=mpimg.imread(name)
                images.append(img)
                images.append(np.fliplr(img)) 
                #left_img
                filename=batch_sample[0][1].split('/')[-1]
                name=batch_sample[1]+'/IMG/'+filename
                img=mpimg.imread(name)
                images.append(img)
                #right image 
                filename=batch_sample[0][2].split('/')[-1]
                name=batch_sample[1]+'/IMG/'+filename
                img=mpimg.imread(name)
                images.append(img)
                
                measurement = float(batch_sample[0][3])
                angles.append(measurement)
                angles.append(-measurement)
                angles.append(measurement+0.2)
                angles.append(measurement-0.2)

            # trim image to only see section with road
            X_train = np.array(images,dtype=np.float32)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model=Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)*IMAGES_PER_SAMPLE//BATCH_SIZE,
validation_data=validation_generator, validation_steps=len(validation_samples)*IMAGES_PER_SAMPLE//BATCH_SIZE, epochs=4)
                    
model.save('model_generator.h5')
print('savingmodel22')



