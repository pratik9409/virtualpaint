import cv2
import numpy as np
import time
import os
import handexample as he


brushThick = 10
eraserThick = 20

folderPath = 'photos'
myList = os.listdir(folderPath)
#print(myList)


overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))
header = overlayList[0]
drawColor = (0,255,0)



cap = cv2.VideoCapture()
address= "https://192.168.0.101:8080/video"
cap.open(address)
cap.set(3, 1024)
cap.set(4,650)


detector = he.handDetector(detectionCon = 0.85)

xp, yp = 0, 0


imgCanvas = np.zeros((650,1024,3), np.uint8)

while True:
    #import image
    success, img = cap.read()
    img = cv2.resize(img,(1024,650))
    img = cv2.flip(img,1)
    
    #Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        
        #print(lmList)
             
        #tip and index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
       
       
       
        #check which fingers are up
        fingers = detector.fingerUp()
        #print(fingers)
       
       
       
        # If selection mode- Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
           
            print("Selection Mode")
            if y1 < 176:
               if 250 < x1 < 385:
                   header = overlayList[0]
                   drawColor = (0,255,0)
               elif 440 < x1 < 580:
                   header = overlayList[1]
                   drawColor = (0,0,255)
               elif 635 < x1 < 765:
                   header = overlayList[2]
                   drawColor = (255,0,0)
               elif 835 < x1 < 960:
                   header = overlayList[3]
                   drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25),drawColor, cv2.FILLED)
           
       #if drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img,(x1,y1), 15, drawColor, cv2.FILLED)
            print('Drawing Mode')
            if xp==0 and yp==0:
                xp, yp = x1, y1
                    
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1), drawColor,eraserThick)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor,eraserThick)
            else:
                cv2.line(img, (xp,yp),(x1,y1), drawColor,brushThick)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor,brushThick)
                
        xp, yp = x1, y1
          
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
   
   
   
    #Setting the header image
    img[0:176, 0:1020, :] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5,0)
    cv2.imshow('Image', img)
    cv2.imshow('Canvas', imgCanvas)
    cv2.imshow('Inverse', imgInv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
