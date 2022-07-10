import cv2
import numpy as np
import features

img = cv2.imread('images/logos/verizon.png')
frame = cv2.imread('images/screenshots/verizon.png')
train_features = features.getFeatures(img)
# detect features on test image
region = features.detectFeatures(frame, train_features)
if region is not None:
    # draw rotated bounding box
    box = cv2.boxPoints(region)
    box = np.int0(box)
    frame1 = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    # display the image
    cv2.imshow("Preview", frame1)
    cv2.waitKey()