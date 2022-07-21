import cv2
import numpy as np
import features



img = cv2.imread('images/logos/verizon.png')
frame = cv2.imread('images/screenshots/verizon.png')

print('Original Dimensions : ',frame.shape)
 
scale_percent = 500 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

train_features = features.getFeatures(img)
# detect features on test image
region = features.detectFeatures(resized, train_features)
if region is not None:
    # draw rotated bounding box
    box = cv2.boxPoints(region)
    box = np.int0(box)
    frame1 = cv2.drawContours(resized, [box], 0, (0, 255, 0), 2)
    # display the image
    cv2.imshow("Preview", frame1)
    cv2.waitKey()