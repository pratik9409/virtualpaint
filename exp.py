import cv2
import mediapipe as mp
import time
import handexample as he




pTime = 0
cTime = 0

cap = cv2.VideoCapture()
address= "https://192.168.0.100:8080/video"
cap.open(address)
detector = he.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist[4])
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,100),2)
        

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
