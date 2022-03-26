'''
CBIR HOG and Bhattacharyya distance
Input: path to an image
Output: 5 most similar images
'''

# import the necessary packages
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


def main():
    
    #Input image
    #ej: rice_leaf_diseases_no_back/leaf smut/leaf_smut_6.jpg (without "")
    image_path = input("Enter image path:")
    
    # Choose if the images' dataset (y is fixed background)
    choose_back = input("Want to use the database with fixed background? (y/n):")
    if choose_back == "y":
        background = "rice_leaf_diseases\\/*\\"
    else:
        background = "rice_leaf_diseases_no_back\\/*\\"

    # Read chosen image
    chosen_image     = getIMG(image_path)
    # Compute HOG of the image
    chosen_image_hog = getHOG(chosen_image)
    
    # Read comparison images
    images_hog  = []
    images_name = []
    
    for image_path in glob.glob(background + "/*.JPG"):
        image = getIMG(image_path)
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

if __name__ == '__main__':
    main()
