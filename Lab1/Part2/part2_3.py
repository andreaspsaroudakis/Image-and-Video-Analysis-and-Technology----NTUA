import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from skimage.util import random_noise
from skimage import filters
from skimage.morphology import disk
from mask import create_mask


# Parameters for Harris/Shi-Tomasi corner detection
feature_params = dict(maxCorners = 150, qualityLevel = 0.01, minDistance = 20, blockSize = 7, useHarrisDetector = True)

# Open video
cap = cv.VideoCapture("video.mp4")

#Read first frame
ret, first_frame = cap.read()

#Resize first frame to half size
first_frame_rs = cv.resize(first_frame,(int(first_frame.shape[1]/2),int(first_frame.shape[0]/2)))

# Converts this first frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
first_frame_gray = cv.cvtColor(first_frame_rs, cv.COLOR_BGR2GRAY)

# Create mask for features extraction and its visualisation on the first frame
mask2D,mask3D = create_mask(first_frame_rs.copy())

# Finds the strongest corners in the first frame by Harris/Shi-Tomasi method - we will track the optical flow for these corners
prev = cv.goodFeaturesToTrack(first_frame_gray, mask = mask2D, **feature_params)

x_coord = [] # list with x-coordinates
y_coord = [] # list with y-coordinates

for i in range(prev.shape[0]):
    x_coord.append(prev[i][0][0])
    y_coord.append(prev[i][0][1])
coords = zip(x_coord,y_coord)

for x, y in coords:
    cv.circle(first_frame_rs,(x,y),3,(0,255,0),-1)

cv.imshow("First_frame_corners",first_frame_rs)
cv.waitKey(10000)

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
