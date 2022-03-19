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


def getHCL(image):
	image_hcl = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	image_hcl = cv2.normalize(image_hcl, image_hcl).flatten()
	return image_hcl


IMAGE_PATH = "rice_leaf_diseases/leaf smut/leaf_smut_6.jpg"

# Read chosen image
chosen_image     = getIMG(IMAGE_PATH)
chosen_image_hcl = getHCL(chosen_image)

# Read comparison images
images_hcl  = []
images_name = []

for image_path in glob.glob("rice_leaf_diseases\\/*\\" + "/*.JPG"):
    
    image = getIMG(image_path)
    images_hcl.append(getHCL(image))
    images_name.append(str(image_path.split(sep="\\")[2]))

distances_hcl = []

for image_hcl in images_hcl:
    distances_hcl.append(cv2.compareHist(image_hcl, chosen_image_hcl, cv2.HISTCMP_BHATTACHARYYA))
    
most_similar_hcls = np.argsort(distances_hcl)[:5]
print(np.array(images_name)[most_similar_hcls])