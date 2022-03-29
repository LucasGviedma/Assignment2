'''
CBIR HOG and Bhattacharyya distance
Input: path to an image
Output: 5 most similar images
'''
# INPUT IMAGE
# ej: rice_leaf_diseases_no_back/leaf smut/leaf_smut_6.jpg
image_path = "rice_leaf_diseases_no_back/leaf smut/leaf_smut_6.jpg"

# Import the necessary packages
import glob
import cv2
import numpy as np
from skimage.io import imread

# Function to pre-process the image
# Input: path to the image
# Output: image
def getIMG(imagePath):
    image = imread(imagePath)
    image = cv2.resize(image, (2398, 721))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to get the HOG of an image
# Input: image
# Output: HOG
def getHOG(image):
	image_hog = cv2.HOGDescriptor().compute(image, (32, 32))
	image_hog = cv2.normalize(image_hog, image_hog).flatten()
	return image_hog

# Read chosen image
chosen_image     = getIMG(image_path)
# Compute HOG of the image
chosen_image_hog = getHOG(chosen_image)
    
# Read comparison images
images      = []
images_hog  = []
images_name = []
    
for image_path in glob.glob("rice_leaf_diseases_no_back\\/*\\" + "/*.JPG"):
    image = getIMG(image_path)
    images.append(image)
    images_hog.append(getHOG(image))
    images_name.append(str(image_path.split(sep="\\")[2]))
    
distances_hog = []
# Compare the distance of the hog images
for image_hog in images_hog:
    distances_hog.append(cv2.compareHist(image_hog, chosen_image_hog, cv2.HISTCMP_BHATTACHARYYA))
    
# Print the 5 images with the lower distance      
most_similar_hogs = np.argsort(distances_hog)[:5]
print("The five most similar images are: \n")
print(np.array(images_name)[most_similar_hogs])

cv2.imshow("INPUT - " + image_path, cv2.resize(chosen_image, (1189,360)))
cv2.imshow("Rank 1 - " + images_name[most_similar_hogs[0]], cv2.resize(images[most_similar_hogs[0]], (1189,360)))
cv2.imshow("Rank 2 - " + images_name[most_similar_hogs[1]], cv2.resize(images[most_similar_hogs[1]], (1189,360)))
cv2.imshow("Rank 3 - " + images_name[most_similar_hogs[2]], cv2.resize(images[most_similar_hogs[2]], (1189,360)))
cv2.imshow("Rank 4 - " + images_name[most_similar_hogs[3]], cv2.resize(images[most_similar_hogs[3]], (1189,360)))
cv2.imshow("Rank 5 - " + images_name[most_similar_hogs[4]], cv2.resize(images[most_similar_hogs[4]], (1189,360)))

cv2.waitKey()