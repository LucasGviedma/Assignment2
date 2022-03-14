# import the necessary packages
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import numpy as np
import argparse
import glob
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn import svm
from skimage import exposure

images = {}

hog_images = []
hog_featur = []

# AVG SIZES -> x = 2398.44 || y = 721.34

# loop over the image paths
for imagePath in glob.glob("rice_leaf_diseases\\/*\\" + "/*.JPG"):

	image = imread(imagePath)
 
	filename = str(imagePath.split(sep="\\")[2])
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
	resized_img = resize(image, (721.34, 2398.44))
	hog_feature, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
 
	hog_images.append(hog_image)
	hog_featur.append(hog_feature)


clf = svm.SVC()
hog_featur = np.array(hog_featur)
np.random.shuffle(hog_featur)

percentage = 80
partition = int(len(hog_featur)*percentage/100)

x_train, x_test = hog_featur[:partition,:-1],  hog_featur[partition:,:-1]
y_train, y_test = hog_featur[:partition,-1:].ravel() , hog_featur[partition:,-1:].ravel()

clf.fit(x_train,y_train)