import numpy as np
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
lines=read_samples('bad_curve_track2/driving_log.csv','bad_curve_track2/',lines) ### center drive in reversed track1
print(len(lines))

images=[]
angles=[]

batch_sample=lines[np.random.randint(0,len(lines))]
filename=batch_sample[0][0].split('/')[-1]
name=batch_sample[1]+'/IMG/'+filename
img=mpimg.imread(name)
images.append(img)
mpimg.imsave('./examples/sharpest_curve_track2/center_image.png',img)

images.append(np.fliplr(img)) 
mpimg.imsave('./examples/sharpest_curve_track2/flipped_center_image.png',np.fliplr(img))

#left_img
filename=batch_sample[0][1].split('/')[-1]
name=batch_sample[1]+'/IMG/'+filename
img=mpimg.imread(name)
images.append(img)
mpimg.imsave('./examples/sharpest_curve_track2/left_image.png',img)

#right image 
filename=batch_sample[0][2].split('/')[-1]
name=batch_sample[1]+'/IMG/'+filename
img=mpimg.imread(name)
images.append(img)
mpimg.imsave('./examples/sharpest_curve_track2/right_image.png',img)
measurement = float(batch_sample[0][3])
angles.append(measurement)
angles.append(-measurement)
angles.append(measurement+0.23)
angles.append(measurement-0.23)

file = open('./examples/sharpest_curve_track2/angles.txt','w') 
file.write('center angle: '+str(measurement)+'\n' )
file.write('flipped angle: '+str(-measurement) +'\n' )
file.write('left angle: '+str(measurement+0.23) +'\n' )
file.write('right angle: '+str(measurement-0.23) +'\n' )
file.close() 

print('angles',angles)
