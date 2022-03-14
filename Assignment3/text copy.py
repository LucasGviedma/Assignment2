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
from skimage import exposure

images = {}
index  = {}
hog_i  = {}

# AVG SIZES -> x = 2398.44 || y = 721.34

# loop over the image paths
for imagePath in glob.glob("rice_leaf_diseases\\/*\\" + "/*.JPG"):

	image = imread(imagePath)
 
	filename = str(imagePath.split(sep="\\")[2])
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
	resized_img = resize(image, (721.34, 2398.44))
	fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
 
	hog_i[filename] = hog_image
	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index

	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	index[filename] = hist

def extract_hog(img):
	# resizing image
	resized_img = resize(img, (128*4, 64*4))
	plt.axis("off")
	plt.imshow(resized_img)
	#creating hog features
	fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
	plt.axis("off")
	plt.imshow(hog_image, cmap="gray")




print(index.keys)
print(images.keys)
 # METHOD #1: UTILIZING OPENCV
# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL),
	("Chi-Squared", cv2.HISTCMP_CHISQR),
	("Intersection", cv2.HISTCMP_INTERSECT),
	("Hellinger", cv2.HISTCMP_BHATTACHARYYA))
# loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
	# initialize the results dictionary and the sort
	# direction
	results = {}
	reverse = False
	# if we are using the correlation or intersection
	# method, then sort the results in reverse order
	if methodName in ("Correlation", "Intersection"):
		reverse = True
	
	# loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = cv2.compareHist(index["doge.png"], hist, method)
		results[k] = d
	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
	
	# show the query image
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images["doge.png"])
	plt.axis("off")
	# initialize the results figure
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)
	# loop over the results
	for (i, (v, k)) in enumerate(results):
		# show the result
		ax = fig.add_subplot(1, len(images), i + 1)
		ax.set_title("%s: %.2f" % (k, v))
		plt.imshow(images[k])
		plt.axis("off")
# show the OpenCV methods
plt.show()