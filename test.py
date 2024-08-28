import cv2
import os

#get current path
current_path = os.getcwd()
# Load the two images
image1 = cv2.imread(os.path.join(current_path , 'cv' , 'image2.jpg'))
image2 = cv2.imread(os.path.join(current_path , 'cv' , 'image1.jpg'))

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Initialize the Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Check if there are enough good matches
good_matches = [m for m in matches if m.distance < 50]

if len(good_matches) > 5:
    print("The objects in the two images are likely the same.")
else:
    print("The objects in the two images are likely different.")