import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

img = cv2.imread(r'C:\Users\Austin\Desktop\Agent\Hackathons\Safari\Train\Mosque\ImageID_1JYS9RN3.jpg')

imgycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

# Masking
mask = cv2.inRange(imgycrcb, min_YCrCb, max_YCrCb)

result = cv2.bitwise_and(img, img, mask=mask)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h = hsv[:,:,0]
# s = hsv[:,:,1]
# v = hsv[:,:,2]
# hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow('Image',mask)
cv2.imshow('Result', result)

contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1,(0,0,255),2)

for c in contours:
        area = cv2.contourArea(c)
        # remove noise- small sections
        if area > 800:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.drawContours(img, c, -1,(0,0,255),2)
            
            cv2.imshow('Result: ', img)


cv2.waitKey(0)
cv2.destroyAllWindows()


#  Read the original image
img = cv2.imread(r'C:\Users\Austin\Desktop\Agent\Hackathons\Safari\Train\Mosque\ImageID_0C44DP1S.jpg')

# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    low_orange = np.array([5,50,50])
    high_orange = np.array([15,255,255])
    low_test = np.array([0,58,30])
    high_rest = np.array([33,255,255])
    
    # YCRCB Color Space  
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)
    
    # Masking
    mask = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Find contours
    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1,(0,0,255),2)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    # b, g, r = cv2.split(result)  
    # filter = g.copy()
    for c in contours:
        area = cv2.contourArea(c)
        # remove noise- small sections
        if area > 400:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.drawContours(frame, c, -1,(0,0,255),2)
            
            cv2.imshow('Result: ', frame)

    # cv2.imshow('Result: ', frame)
    cv2.imshow('Mask: ',mask)
    cv2.imshow('Mask:, ',result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()