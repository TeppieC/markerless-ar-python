'''
Author: Zhaorui

Code based on OpenCV doc - python tutorial.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv2.imread('1.jpg')          # queryImage
img2 = cv2.imread('2.jpg') # trainImage

#img1 = cv2.cvtColor(img1raw, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img2raw, cv2.COLOR_BGR2GRAY)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class Match:
	def __init__(self, roi, frame):
		self.roi = roi
		self.frame = frame

	def getCorrespondence(self):
		self.
''' Matching descriptors '''

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT

kp1, des1 = sift.detectAndCompute(img1,None) # kp1: keypoints from the model
kp2, des2 = sift.detectAndCompute(img2,None) # kp2: keypoints from the frame/captured image
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# simple nn, can be refined with cross validation
# des1: descriptors from the model
# des2: descriptors from the scene
matches = flann.knnMatch(des1,des2,k=2) # [ ..., [<DMatch 0x104a34c30>, <DMatch 0x104a34c50>], ... ]
#print(matches) 

# store all the good matches as per Lowe's ratio test.
good = [] # good is a container for 3D points found in scene
for m,n in matches:
	if m.distance < 0.7*n.distance:
		good.append(m) # m is from the scene/captured image <DMatch 0x104a34c30>
		#print(m)

print(len(good))
if len(good)>MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) # 3d vectors ? How to get the 3D points??
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) # 2d vectors
	print(np.float32([ kp1[m.queryIdx].pt for m in good ]))
	print(np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2))
	#print(src_pts[0])
	#print(dst_pts[0])
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	h,w,d = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)
	img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
	print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
	matchesMask = None
'''
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#cv2.imshow('img3',img3)
#plt.show()
plt.imshow(img3, 'gray')
plt.show()
'''


''' Find out the 2D/3D correspondences '''


