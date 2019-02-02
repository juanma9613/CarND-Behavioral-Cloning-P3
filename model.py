import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Input, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import cv2
from sklearn.utils import shuffle
BATCH_SIZE=32
IMAGES_PER_SAMPLE=4
lines=[]

def read_samples(filepath, folder_name,sample_list):
    #INPUTS FILEPATH (STRING): It's the path to the driving 
    #log file of some data.
    #INPUTS folder_name (STRING): It's the name of the folder which contains
    #the driving log file and the IMG folder..
    #INPUT AND OUTPUT sample_list: A LIST OF TUPLES WHICH
    #CONTAINS ALL THE LINES OF DRIVING_LOGS FROM DIFERENT FOLDERS 
    #AND ALSO THE NAME OF THE FOLDER WHERE THE DATA IS.
	initial_index=len(sample_list)
	with open(filepath) as csvfile:
		reader=csv.reader(csvfile)
		for line in reader:
			sample_list.append([line,folder_name])
	del sample_list[initial_index]
	return sample_list

lines=read_samples('data/driving_log.csv','data/',lines) ### center drive in track1
lines=read_samples('data_recover/driving_log.csv','data_recover/',lines) ### center drive in reversed track1
lines=read_samples('data_track2/driving_log.csv','data_track2/',lines) ### center drive in track2
lines=read_samples('data_track2/driving_log.csv','data_track2/',lines)### center drive in track2
lines=read_samples('bad_curve_track1/driving_log.csv','bad_curve_track1/',lines) ## recovering from the sharp curve in track 1 
lines=read_samples('bad_curve_track2/driving_log.csv','bad_curve_track2/',lines) ## recovering from some sharp curves in track 2 
# this directory was not included to avoid bad behavior
#lines=read_samples('bad_curves_track2_2/driving_log.csv','bad_curves_track2_2/',lines)
lines=read_samples('bad_curves_22/driving_log.csv','bad_curves_22/',lines) ## recovering from the sharpest curve in track 2 

print('total samples in folders:',len(lines))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=BATCH_SIZE, steps_per_sample=4):
    ## this generator routine yields # BATCH_SIZE samples every time it's called, to do that, it reads BATCH_SIZE//steps_per_sample
    #lines, and process the corresponding images
    ## INPUT samples: is the output of the read_samples function
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
                angles.append(measurement+0.23)
                angles.append(measurement-0.23)

            # trim image to only see section with road
            X_train = np.array(images,dtype=np.float32)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
            

## input images are 160*320*3
model2=Sequential()
model2.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(160,320,3)))
model2.add(Lambda(lambda x: x/255.0 -0.5)) # Preprocessing
temporal_model=VGG16(weights='imagenet',include_top=False,input_shape=(100,320,3))
## adding the bottom layers of VGG16 to the new model
for layer in temporal_model.layers[:-4]:
	model2.add(layer)
print('model2',model2.summary())
for layer in model2.layers:
	layer.trainable=False
model2.add(Convolution2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block5_conv1'))
model2.add(Convolution2D(512, (3, 3), strides=(1, 2), activation='relu', padding='same', name='block5_conv2'))
model2.add(Convolution2D(512, (3, 3), strides=(1, 2), activation='relu', padding='same', name='block5_conv3'))
model2.add(Flatten())
model2.add(Dropout(.2))
model2.add(Dense(2048, activation='relu', name='fc1'))
model2.add(Dropout(.3))
model2.add(Dense(1024, activation='relu', name='fc2'))
model2.add(Dropout(.5))
model2.add(Dense(1, activation='linear', name='predictions'))
model2.compile(loss='mse',optimizer='adam')

# uncommente this line if you want changer to be a new Keras callback
#changer = Trainable_changer()

checkpoint = ModelCheckpoint('model_ch.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
model2.fit_generator(train_generator, steps_per_epoch= len(train_samples)*IMAGES_PER_SAMPLE//BATCH_SIZE,
validation_data=validation_generator, validation_steps=len(validation_samples)*IMAGES_PER_SAMPLE//BATCH_SIZE, epochs=8,callbacks=[checkpoint, early_stopping])
print(model2.summary())
model2.save('model.h5')



