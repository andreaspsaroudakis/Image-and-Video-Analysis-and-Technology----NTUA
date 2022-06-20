import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from skimage.util import random_noise
from skimage import filters
from skimage.morphology import disk
from mask import create_mask

def snpAmount(x):
  return x/90 + 0.3

def OpticalFlow(video_path,feature_params,lk_params,feature_update_frames,points_distance,line_length,color):

    # Open video
    cap = cv.VideoCapture(video_path)

    #Read first frame
    ret, first_frame = cap.read()

    #Resize first frame to half size
    first_frame_rs = cv.resize(first_frame,(int(first_frame.shape[1]/2),int(first_frame.shape[0]/2)))

    # Converts this first frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame_rs, cv.COLOR_BGR2GRAY)

    # Add s&p noise to frame and convert the result to range [0,255] instead of [0,1]
    prev_gray = (random_noise(prev_gray, mode='s&p', seed=9, amount = snpAmount(7))) * 255

    # Convert the noisy frame from type float to type uint8 (necessary for function calcOpticalFlowPyrLK)
    prev_gray = prev_gray.astype(np.uint8)

    # Create mask for features extraction and its visualisation on the first frame 
    mask2D,mask3D = create_mask(first_frame_rs.copy())

    # Finds the strongest corners in the first frame by Harris/Shi-Tomasi method - we will track the optical flow for these corners
    prev = cv.goodFeaturesToTrack(prev_gray, mask = mask2D, **feature_params)

    # frame counter
    frames = 0

    #queue for masks
    mask_queue = []

    while(cap.isOpened()):

        # Increase frame counter
        frames = frames + 1
         
        # Read next frame 
        ret, frame = cap.read()

        # Resize frame to half size 
        frame_rs = cv.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

        # Convert frame to grayscale
        gray = cv.cvtColor(frame_rs, cv.COLOR_BGR2GRAY)

        # Add s&p noise to frame and convert the result to range [0,255] instead of [0,1]
        gray = (random_noise(gray, mode='s&p', seed=9, amount = snpAmount(7))) * 255

        # Convert the noisy frame from type float to type uint8 (necessary for function calcOpticalFlowPyrLK)
        gray = gray.astype(np.uint8)

        # Calculate sparse optical flow by Lucas-Kanade method
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

        # Select good feature points for previous position
        good_old = prev[status == 1]

        # Select good feature points for next position
        good_new = next[status == 1]

        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        mask = np.zeros_like(first_frame_rs)

        # Apply logical "or" operation between all the masks in mask_queue
        for masks in mask_queue: mask = cv.bitwise_or(mask,masks)

        # Check the size of the mask_queue
        if len(mask_queue) > line_length: mask_queue.pop(0)
    
        # Creates a temporary image filled with zero intensities with the same dimensions as the frame - we will draw current optical flow on it
        mask_temp = np.zeros_like(first_frame_rs)

        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):

            # a, b = coordinates of new point
            a, b = new.ravel()

            # c, d = coordinates of old point
            c, d = old.ravel()

             # Draws line between new and old position with green color and 2 thickness
            if  distance.euclidean(new,old) > points_distance : 
                
                # Draws line between new and old position with green color and 2 thickness 
                mask = cv.line(mask, (a, b), (c, d), color, 2) 

                # Draw the same line on the temporary mask
                mask_temp = cv.line(mask_temp, (a, b), (c, d), color, 2)

                # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                frame_rs = cv.circle(frame_rs, (a, b), 2, color, -1)

        # Append the temporary mask in the mask_queue
        mask_queue.append(mask_temp)

        # Overlays the optical flow tracks on the original frame
        output = cv.add(frame_rs, mask)

        # Updates previous frame
        prev_gray = gray.copy()

        # Updates feature points
        if frames==feature_update_frames:
            prev =  cv.goodFeaturesToTrack(gray, mask = mask2D, **feature_params)
            frames=0
        else:
            prev = good_new.reshape(-1, 1, 2)

        # Opens a new window and displays the output frame
        cv.imshow("sparse optical flow", output)

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    
    # Video path
    video_path = "video.mp4"

    # Parameters for Harris/Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 150, qualityLevel = 0.12, minDistance = 20, blockSize = 7, useHarrisDetector = False)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (15,15), maxLevel = 4, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Defines how often we update the features in the video
    feature_update_frames = 25

    # Distance between points in two successive images
    points_distance = 0.1

    # Color for the optical flow in BGR plane
    color = (0, 255, 0)

    # Defines how long we wish the optical flow lines to be
    line_length = 50

    # Call the function that 
    OpticalFlow(video_path,feature_params,lk_params,feature_update_frames,points_distance,line_length,color)
