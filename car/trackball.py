import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import statistics

low_bound= np.array([0,150, 170])
high_bound= np.array([180,255, 255])
sizeBall = 4.0

def main():
    #opens the webcam
    cap = cv2.VideoCapture(0)

    #set params and creates blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.filterByConvexity= False
    params.filterByCircularity= False
    params.filterByInertia= False
    params.filterByArea = False
    params.blobColor = 255

    blob_detector = cv2.SimpleBlobDetector_create(params)
    
    #remember important stuff
    all_keypoints = []
    frames = []
    threshold_frames = []
    times = []

    #set kernels
    kernel_open = np.ones((5,5,), np.uint8)
    kernel_close = np.ones((25,25,), np.uint8)

    #set timer
    start = time.perf_counter()

    while(True):
        _, imgOriginal = cap.read();
        
        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
        imgThreshold = cv2.inRange(imgHSV, low_bound, high_bound) 
     
        #morph opening
        imgThreshold = cv2.erode(imgThreshold, kernel_open, iterations=1);
        imgThreshold = cv2.dilate(imgThreshold, kernel_open, iterations=1);
        #morph closing
        imgThreshold = cv2.dilate(imgThreshold, kernel_close, iterations=1);
        imgThreshold = cv2.erode(imgThreshold, kernel_close, iterations=1);

        keypoints = blob_detector.detect(imgThreshold);
        if(len(keypoints) ==1):
            all_keypoints+=keypoints
            frames.append(imgOriginal)
            times.append(time.perf_counter())
            threshold_frames.append(imgThreshold)

        imgOriginal = cv2.drawKeypoints(imgOriginal, keypoints, np.array([]), [0,0,255], cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS) 

        cv2.imshow("Original", imgOriginal)
        cv2.imshow("Threshold", imgThreshold)

        if (cv2.waitKey(30) ==27):
            break

    #average blal size
    ballsize2 = (sum([p.size for p in all_keypoints]) / (sizeBall* len(all_keypoints)))
    ballsize = statistics.median([p.size for p in all_keypoints]) /sizeBall
    print(ballsize)
    print(ballsize2)

    #3 offset times for regression
    start = times[0]
    times[:] = [x - start for x in times]
    
    position = [(p.pt[1] / ballsize) for p in all_keypoints]

    print(position)
    print(times)

    position_reg = np.array(position)
    times_reg = np.array(times)
    coeff = np.polyfit(times_reg, position_reg, 2)

    print(coeff)
    print(str(coeff[0]/50) + " m/s^2")

    i=0 #display stuff and frames
    while(i<len(frames)):
        cv2.destroyAllWindows()
        cv2.imshow("Frame " + str(i), frames[i])
        cv2.imshow("Threshold " + str(i), threshold_frames[i])
        a = cv2.waitKey(0)
        if(a == 110):
            i +=1
        elif(a ==27):
            break

main()


    
