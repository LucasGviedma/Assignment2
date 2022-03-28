'''
CBIR HCL and Bhattacharyya distance
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

# Function to get the histogram of color of an image
# Input: image
# Output: HCL
def getHCL(image):
	image_hcl = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	image_hcl = cv2.normalize(image_hcl, image_hcl).flatten()
	return image_hcl

# Read chosen image
chosen_image     = getIMG(image_path)
# Compute HCL of the image
chosen_image_hcl = getHCL(chosen_image)
    
# Read comparison images
images_hcl  = []
images_name = []

for image_path in glob.glob("rice_leaf_diseases_no_back\\/*\\" + "/*.JPG"):
    image = getIMG(image_path)
    images_hcl.append(getHCL(image))
    images_name.append(str(image_path.split(sep="\\")[2]))

distances_hcl = []
# Compare the distance of the hcl images
for image_hcl in images_hcl:
    distances_hcl.append(cv2.compareHist(image_hcl, chosen_image_hcl, cv2.HISTCMP_BHATTACHARYYA))
     
# Print the 5 images with the lower distance 
most_similar_hcls = np.argsort(distances_hcl)[:5]
print("The five most similar images are:")
print(np.array(images_name)[most_similar_hcls])
