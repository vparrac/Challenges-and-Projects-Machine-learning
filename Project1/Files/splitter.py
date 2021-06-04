import cv2
import numpy as np
import os

cap = cv2.VideoCapture('Super Mario Bros. (1985) Full Walkthrough NES Gameplay [Nostalgia].mp4')

try:
    if not os.path.exists('data/video2'):
        os.makedirs('data/video2')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    ret, frame = cap.read()
    name = './data/video2/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)
    currentFrame += 1

cap.release()
cv2.destroyAllWindows() 