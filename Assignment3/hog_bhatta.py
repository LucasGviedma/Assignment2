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


IMAGE_PATH = "rice_leaf_diseases_no_back/leaf smut/leaf_smut_6.jpg"

# Read chosen image
chosen_image     = getIMG(IMAGE_PATH)
chosen_image_hog = getHOG(chosen_image)

# Read comparison images
images_hog  = []
images_name = []

for image_path in glob.glob("rice_leaf_diseases_no_back\\/*\\" + "/*.JPG"):
    
    image = getIMG(image_path)
    images_hog.append(getHOG(image))
    images_name.append(str(image_path.split(sep="\\")[2]))

distances_hog = []

for image_hog in images_hog:
    distances_hog.append(cv2.compareHist(image_hog, chosen_image_hog, cv2.HISTCMP_BHATTACHARYYA))
    
most_similar_hogs = np.argsort(distances_hog)[:5]
print(np.array(images_name)[most_similar_hogs])