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
BATCH_SIZE=28
IMAGES_PER_SAMPLE=4
lines=[]

def read_samples(filepath, folder_name,sample_list):
	initial_index=len(sample_list)
	with open(filepath) as csvfile:
		reader=csv.reader(csvfile)
		for line in reader:
			sample_list.append([line,folder_name])
	del sample_list[initial_index]
	return sample_list

        
lines=read_samples('bad_2/driving_log.csv','bad_2/',lines)
lines=read_samples('bad_curves_22/driving_log.csv','bad_curves_22/',lines)


print('total samples in folders:',len(lines))
#print('headers',lines[0])




from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.3)

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
                angles.append(measurement+0.23)
                angles.append(measurement-0.23)

            # trim image to only see section with road
            X_train = np.array(images,dtype=np.float32)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
            
class Trainable_changer(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.reminder = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch >=1:
            for layer in self.model.layers:
                if layer.name[:6] == 'block4':
                    print('layer changed')
                    layer.trainable=True
    #def on_batch_end(self, batch, logs={}):
        #print('batch_end :',batch,'\n')
                  
       
   



model2=keras.models.load_model('transfer_best_23_track2_curves.h5')
checkpoint = ModelCheckpoint('transfer_best_23_track2_curves_new.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
model2.fit_generator(train_generator, steps_per_epoch= len(train_samples)*IMAGES_PER_SAMPLE//BATCH_SIZE,
validation_data=validation_generator, validation_steps=len(validation_samples)*IMAGES_PER_SAMPLE//BATCH_SIZE, epochs=8,callbacks=[checkpoint, early_stopping])
model2.save('retrain.h5')



