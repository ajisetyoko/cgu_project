# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-02-25T16:23:49+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-02-25T16:38:26+08:00



import numpy as np
import cv2
import sys

rect = (0,0,0,0)
startPoint = False
endPoint = False

def on_mouse(event,x,y,flags,params):

    global rect,startPoint,endPoint

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:
            rect = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True

video_file = sys.argv[1]
print(video_file)
cap = cv2.VideoCapture(video_file)

waitTime = 50

#Reading the first frame
(grabbed, frame) = cap.read()

while(cap.isOpened()):

    ret_val, frame = cap.read()
    if ret_val == True:
        cv2.namedWindow('main',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('main', on_mouse)

        #drawing rectangle
        if startPoint == True and endPoint == True:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
            print('Point',rect[0], rect[1],rect[2], rect[3])


        cv2.imshow('main',frame)
        cv2.resizeWindow('main', (480*2,270*2))

        key = cv2.waitKey(waitTime)

        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
