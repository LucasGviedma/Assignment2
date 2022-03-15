# import the necessary packages
import glob
import cv2
import scipy.spatial.distance as dist
import numpy as np
from skimage.io import imread


def getIMG(imagePath):
    image = imread(imagePath)
    image = cv2.resize(image, (2398, 721))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def getHOG(image):
	image_hog = cv2.HOGDescriptor().compute(image, (32, 32))
	image_hog = cv2.normalize(image_hog, image_hog).flatten()
	return image_hog


def getHCL(image):
	image_hcl = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	image_hcl = cv2.normalize(image_hcl, image_hcl).flatten()
	return image_hcl


# Read chosen image
chosen_image     = getIMG("rice_leaf_diseases/leaf smut/leaf_smut_6.jpg")
chosen_image_hog = getHOG(chosen_image)
chosen_image_hcl = getHCL(chosen_image)

# Read comparison images
images     = []
images_nam = []
images_hog = []
images_hcl = []

for image_path in glob.glob("rice_leaf_diseases\\/*\\" + "/*.JPG"):
    
    image = getIMG(image_path)
    images.append(image)
    images_hog.append(getHOG(image))
    images_nam.append(str(image_path.split(sep="\\")[2]))
    
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #new_values = [[pixel if pixel < 225 else 0 for pixel in row] for row in hsv_image[:,:,2]]
    #hsv_image[:,:,2] = new_values

distances_hog = []
distances_hcl = []

for image_hog in images_hog:
    distances_hog.append(dist.euclidean(image_hog, chosen_image_hog))
most_similar_hogs = np.argsort(distances_hog)[:15]

for i in np.array(images)[most_similar_hogs]:
	images_hcl.append(getHCL(image))

for image_hcl in images_hcl:
    distances_hcl.append(dist.euclidean(image_hcl, chosen_image_hcl))
most_similar_hcls = np.argsort(distances_hcl)[:5]

most_similar_imgs = np.array(most_similar_hogs)[most_similar_hcls]
print(np.array(images_nam)[most_similar_imgs])
