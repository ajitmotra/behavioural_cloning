import csv
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn import preprocessing
from keras.optimizers import SGD
from keras.optimizers import Adam
import hashlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import math
from skimage import color
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2


#####  ##### 
batch_size = 4
re_size = 160, 60
###
X_train = []
y_train_ang = []
img_rows = re_size[1]
img_cols = re_size[0]
learning_rate = 0.00001

def normalize_image(image_data):    
    mean = np.mean(image_data)
    image_data = image_data - mean
    manima = np.min(image_data)
    Scaled_image = image_data - manima
    maxima = np.max(Scaled_image)
    Scaled_image = Scaled_image/maxima
    Scaled_image = Scaled_image - 0.5
    return Scaled_image

def generate_arrays_from_file(path):
		while 1:
			csv_file = open("./simulator-linux/driving_log.csv")
			fileread = csv.reader(csv_file)
			l = list(fileread)
			random.shuffle(l)
			for row in l:
				X_train = []
				y_train = []
				#center	
				image = Image.open(row[0])
				image.load()	
				image = image.crop((0,40, 320, 160))
				image.thumbnail(re_size, Image.ANTIALIAS)
				image1 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
				#image1[0] = cv2.equalizeHist(image1[0])
				#image1 = cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=5)
				image.close()
				#left
				image = Image.open(row[1])
				image.load()	
				image = image.crop((0,40, 320, 160))
				image.thumbnail(re_size, Image.ANTIALIAS)
				image2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
				#image2[0] = cv2.equalizeHist(image2[0])
				#image2 = cv2.Sobel(image2,cv2.CV_64F,1,0,ksize=5)
				image.close()
				#Right
				image = Image.open(row[2])
				image.load()	
				image = image.crop((0,40, 320, 160))
				image.thumbnail(re_size, Image.ANTIALIAS)
				image3 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
				#image3[0] = cv2.equalizeHist(image3[0])
				#image3 = cv2.Sobel(image3,cv2.CV_64F,1,0,ksize=5)
				image.close()
				#image1 = normalize_image(np.array(image1))
				#print(float(row[3]))
				#plt.imshow(image1)							
				#plt.show()
				
				X_train.append(np.array(image2))  # Left 	
				y_train.append(float(row[3]) + 0.25)

				if(abs (float(row[3])) > 0.01):	
					X_train.append(np.array(image3))  # Right 	
					y_train.append(float(row[3]) - 0.25)

				if(abs (float(row[3])) > 0.01):
					X_train.append(np.array(image1))  # center 	
					y_train.append(float(row[3]))	
				
				if(float(row[3]) > 0.1):					
					X_train.append(np.fliplr(np.array(image1)))  # flipped
					y_train.append(float(row[3])*-1.0)

				if(float(row[3]) < -0.1):					
					X_train.append(np.fliplr(np.array(image1)))  # flipped
					y_train.append(float(row[3])*-1.0)				 				
				X_train = np.array(X_train)
				y_train = np.array(y_train, dtype=float)
				if(len(X_train)):
					yield (X_train, y_train)				
			csv_file.close()

# load json and create model
json_file = open('model_working/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_working/model.h5")
print("Loaded model from disk")

adam = Adam(lr=learning_rate)

model.compile(loss="mse",
              optimizer=adam)

model.fit_generator(generate_arrays_from_file('./driving_log.csv'),
        samples_per_epoch=15000, nb_epoch=5, verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print("Saved Model to the disk")
