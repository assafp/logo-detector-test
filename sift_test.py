import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img2 = cv.imread('images/screenshots/fortinet.png')#,cv.IMREAD_GRAYSCALE)          # queryImage
img1 = cv.imread('images/logos/fortinet.png')#,cv.IMREAD_GRAYSCALE) # trainImage 
img1 = cv.bitwise_not(img1)
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# plt.imshow(img2,),plt.show()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
good = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good.append([m])

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()



# estimate a transformation matrix which maps keypoints from train image coordinates to sample image
src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good
                      ]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good
                      ]).reshape(-1, 1, 2)

m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)



shape = img1.shape

scene_points = cv.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1),
                                                    (shape[1] - 1, shape[0] - 1),
                                                    (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)


rect = cv.minAreaRect(scene_points)

# import pdb; pdb.set_trace()
box = cv.boxPoints(rect)
box = np.int0(box)


img4 = cv.drawContours(img2, [box], 0, (0, 255, 0), 2)


plt.imshow(img4,),plt.show()