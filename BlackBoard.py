from numpy.core.fromnumeric import trace
import cv2
import numpy as np
import copy
import math

# getPoints function takes two points and return all the points in a list
# which lies on the line passing through those points and 
# between the points
def getPoints(x1, y1,x2,y2):
    l=[]
    m=((y1-y2)/(x1-x2))
    c= (y2*x1- y1*x2)/(x1-x2)
    for x in range(x1,x2):
        y= x*m + c
        l.append([x,int(y)])
    l.append([x2,int(y2)])
    return l

# this is helper function that prints if threshold is changed
def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

# get the centroid of the contour using moment based on Intensity of each pixel   
def centroid(contour):
    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

# it takes a frame and remove the background intially captured
def removeBackGround(frame):
    foreGroundMask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    foreGroundMask = cv2.erode(foreGroundMask, kernel, iterations=1)
    subtractedFrame = cv2.bitwise_and(frame, frame, mask=foreGroundMask)
    return subtractedFrame

# calculate farthest points of a contour based on defects and centroid
def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None
    else:
        return None
    
#  get count of fingers in a frame  based on angle between defect
def calculateFingers(frame,drawing): 
    #  get convexity defect of frame
    hull = cv2.convexHull(frame, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(frame, hull)
        if type(defects) != type(None):  
            cnt = 0
             # calculate the angle
            for i in range(defects.shape[0]): 
                s, e, f, d = defects[i][0]
                start = tuple(frame[s][0])
                end = tuple(frame[e][0])
                far = tuple(frame[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  
                # angle less than 90 degree, treat as fingers
                if angle <= math.pi / 2: 
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
current_blackboard=np.zeros((600,600,3))
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


# variable that shows whether the background is captured or not
isBgCaptured = 0   

# variable that controls keyborad simulator
triggerSwitch = False  

# different color of cursor
color=[(255,255,255),   # white
       (0,255,0),       # green
       (0,0,255),       # blue
       (255,0,0)]       # red

# initial color of cursor 
current_color=0

# variable that show edit mode of blackboard
edit_mode=1

# variable that control color change 
can_change_color=0

# queue that stores two current points 
queue=[]

# initial text on blackboard
text= "edit"

# postion of text box in blackboard
pos=(520,20)

# continue capture the frame while camera is opened
while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    
    # apply smoothing filter on the capture frame
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  
    
    #flip the captured frame horizontally
    frame = cv2.flip(frame, 1)  
    
    # flip the frame of blackboard horizontally
    flip_blackboard = cv2.flip(current_blackboard, 1) 
    
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    
    cv2.imshow('original', frame)

    #  if background is captured, then start the blackboard for drawing
    if isBgCaptured == 1:  
        
        # remove the background from captured frame
        img = removeBackGround(frame)
        
        # get region of interest from capture frame
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  

        # convert the image into binary frame 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # apply gaussian blur on the frame
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        
        # apply binarization on the frame
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        # now our frame is ready for processing
        thresh1 = copy.deepcopy(thresh)
        
        # find contours of the frame
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        
        # if the number of countors found more than equal to 1 ,
        # if some thing is capture on screen then process it further
        if length > 0:
            
            # find the biggest contour area wise
            for i in range(length):  
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            max_contour = contours[ci]
            
            # get hull of that biggest contour
            hull = cv2.convexHull(max_contour )
            
            # get of center of the biggest contour (hand)
            centre_of_hand = centroid(max_contour)
            
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [max_contour], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            
            isFinishCal,finger_defect_count = calculateFingers(max_contour,drawing)
            
            # if three fingers are shown then we will get two defects 
            # change the blackboard to write mode    
            if finger_defect_count == 2:
                edit_mode=1
            
            # if five fingers are shown then we get four defects
            # change the blackboard to erase mode
            if finger_defect_count == 4:
                edit_mode=0
            
            # if four fingers are shown then toggle current color to next    
            if finger_defect_count ==3 and can_change_color:
                current_color=(current_color+1)%4
                print("toggle")
                can_change_color=0
                
            if finger_defect_count==0:
                can_change_color=1
            
            
            # set returnPoints to false to get defect points also
            hull = cv2.convexHull(max_contour, returnPoints=False)
        
            if len(hull) > 3:
                
                # get the defect of the hull 
                defects = cv2.convexityDefects(max_contour,hull)
                
                # get the farthest point and cosider it as finger tip
                fingerTip = farthest_point(defects, max_contour, centre_of_hand)
                
                # draw a circle on drawing frame on finger tip
                cv2.circle(drawing, fingerTip, 10, [255, 255, 0], -1)
                
                # reset the text section in blackboard frame
                flip_blackboard[0:20, 580:600]=(0.0,0.0,0.0)
            
                # if a finger is captured in webcam  
                # then we have either edit mode or erase mode
                if fingerTip is not None:
                    
                    if edit_mode==1:
                        print("edit")
                        text="edit"
                        
                        queue.append([fingerTip[0],fingerTip[1]])
                        
                        # set finger point on blackboard to the current color
                        flip_blackboard[fingerTip[0]-5:fingerTip[0]+5,fingerTip[1]-5:fingerTip[1]+5]=color[current_color]
                        
                        # get all the points of two continously captured close points and 
                        # color all those points to get smooth drawing experience
                        if len(queue)==2:
                            x1,y1=queue[0]
                            x2,y2=queue[1]
                            
                            # get all the points between current two finger points
                            p=getPoints(min(x1,x2),min(y1,y2),max(x2,x1),max(y2,y1))
                            
                            # if manhattan distance between the points is more than 30
                            # means it is a jump so don't color in between points
                            # else color them
                            if(len(p)<30):
                                for i in range(len(p)):
                                    flip_blackboard[p[i][0]-5:p[i][0]+5,p[i][1]-5:p[i][1]+5]=color[current_color]

                            # pop the first point 
                            queue.pop(0)
                    
                    else :
                        text="erase"
                        print("erase")
                        flip_blackboard[fingerTip[0]-15:fingerTip[0]+15,fingerTip[1]-15:fingerTip[1]+15]=(0.0,0.0,0.0)
                
                # if fist is shown then move the cursor accorording to fist position    
                else:
                    queue=[]
                    text="move"
                   
        blackboard=cv2.transpose(flip_blackboard)
        
        # set the text area to black in the blackboard frame
        blackboard[550:50,0:50]=(0.0,0.0,0.0)
        
        # put the text in the text are of the black board frame
        cv2.putText(img=blackboard, text=text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),thickness=1,bottomLeftOrigin = False)

        cv2.imshow('output', drawing)
        cv2.imshow('Finger-Motion-Based-Black-Board', blackboard)
        
        current_blackboard = cv2.flip(flip_blackboard, 1) 
        
    # keyboard operation to caputre background and close the application
    key = cv2.waitKey(10)
    
    # to close the application press 'ESC' key
    if key == 27:  
        camera.release()
        cv2.destroyAllWindows()
        break
    
    # to capture the background press 'b' key
    elif key == ord('b'): 
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '  Background is successfully captured ')

   