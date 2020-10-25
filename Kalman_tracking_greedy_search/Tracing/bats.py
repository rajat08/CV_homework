import pandas as pd
import numpy as np
import cv2 as cv
import math
import glob
#from GreedyTracker import Tracker
from MyTracker import Tracker

def detect(frame,org):
    centroid = []
    contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    CNTS = []
    for c in contours:
        if cv.contourArea(c) > 20:
            CNTS.append(c)

    for cnt in CNTS:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(org, (x, y), (x + w, y + h), (255, 0, 0), 2)
        M = cv.moments(cnt)
        x_bar = int(M['m10'] / M['m00'])
        y_bar = int(M['m01'] / M['m00'])
        tup = (x_bar,y_bar)
        cv.drawMarker(org,tup,(0,255,255),cv.MARKER_CROSS,markerSize=20)
        c = np.array([[x_bar],[y_bar]])
        centroid.append(c)

    cv.imshow("show", org)
    return centroid


def greyscale(frame,background):
    #cv.imshow("frame",frame)
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    sub = background.apply(frame)
    _, sub = cv.threshold(sub, 200, 255, cv.THRESH_BINARY)
    cv.imshow("sub",sub)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return sub


def main():
    filenames = glob.glob("CS585-BatImages/Gray/*.ppm")
    filenames.sort()
    images = [cv.imread(img) for img in filenames]
    background = cv.createBackgroundSubtractorMOG2()
    tracker = Tracker(60,5,10,1)
    track_colors = [(255,255,255),(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127),(225,255,255),(51,255,51),
                    (204,204,255),(102,0,204),(255,204,229)]

    count = 0

    video = []
    for items in range(len(images)):

        frame = images[count]
        frame = cv.resize(frame,(500,500))
        count = count + 1
        grey = greyscale(frame,background)
        img2 = np.zeros((np.array(frame).shape[0], np.array(frame).shape[1], 3))
        img2[:,:,0] = grey
        img2[:, :, 1] = grey
        img2[:, :, 2] = grey

        centroid = detect(grey,frame)

        #if (len(centroid)>2):
        tracker.Update(centroid)
        for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace) - 1):
                        #print(i)
                        #print(j)
                        # Draw trace line
                        #print(tracker.tracks[i].trace[j+1][0][0][0])
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j + 1][0][0]
                        y2 = tracker.tracks[i].trace[j + 1][1][0]
                        clr = tracker.tracks[i].track_id % 15
                        # print(clr)
                        cv.line(img2, (int(x1), int(y1)), (int(x2), int(y2)),track_colors[clr], 1)

                        #cv.circle(frame,(int(x1), int(y1)),2,track_colors[clr],4)

        cv.imshow('Tracking', img2)
        video.append(np.uint8(img2))


        #cv.waitKey(50)

        # Check for key strokes
        k = cv.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    #print(len(video))
    video_write = cv.VideoWriter("hgcb.avi", cv.VideoWriter_fourcc(*'DIVX'), 4, (500, 500))

    for i in range(len(video)):
        video_write.write(video[i])
    video_write.release()



if __name__ =="__main__":
    main()

