'''
Author: Zhaorui

Code based on OpenCV doc - python tutorial.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from Frame import Frame

MIN_MATCH_COUNT = 10
#img1 = cv2.cvtColor(img1raw, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img2raw, cv2.COLOR_BGR2GRAY)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class Matcher:
	def __init__(self, roi, alg):
		self.roi = roi # a ROI pattern object
		self.alg = alg
		#self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # a frame
		#self.frame = frame # a frame object
		#self.pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

	def setFrame(self, frame):
		self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	def getCorrespondence(self):
		# Initiate SIFT detector
		#sift = cv2.xfeatures2d.SIFT_create()
		# find the keypoints and descriptors with SIFT

		#kp1, des1 = sift.detectAndCompute(self.pattern, None)
		#print(des1)
		kp1 = self.roi.keypoints
		des1 = self.roi.descriptors

		#print(len(kp1))
		print('# des for marker', len(des1))
		#kp1, des1 = sift.detectAndCompute(img1,None) # kp1: keypoints from the model
		if self.alg == 'orb':
			orb = cv2.ORB_create()
			kp2, des2 = orb.detectAndCompute(self.frame, None) # kp2: keypoints from the frame/captured image
			FLANN_INDEX_LSH = 6
			index_params= dict(algorithm = FLANN_INDEX_LSH,
				   table_number = 6, # 12
				   key_size = 12,     # 20
				   multi_probe_level = 1) #2
		elif self.alg == 'sift':
			FLANN_INDEX_KDTREE = 1
			sift = cv2.xfeatures2d.SIFT_create()
			kp2, des2 = sift.detectAndCompute(self.frame, None) # kp2: keypoints from the frame/captured image
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

		print('# des for frame', len(des2))
		#print(self.frame.shape) # TODO: no kp/des?

		#kp2 = self.frame.keypoints
		#des2 = self.frame.descriptors
		#print(len(kp2))


		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		# simple nn, can be refined with cross validation
		# des1: descriptors from the model
		# des2: descriptors from the scene
		matches = flann.knnMatch(des1,des2,k=2) # [ ..., [<DMatch 0x104a34c30>, <DMatch 0x104a34c50>], ... ]
		#print(matches) 

		# store all the good matches as per Lowe's ratio test.
		good = [] # good is a container for 3D points found in scene
		for m_n in matches:
			if len(m_n)!=2:
				continue
			# http://stackoverflow.com/questions/25018423/opencv-python-error-when-using-orb-images-feature-matching
			(m, n) = m_n
			if m.distance < 0.7*n.distance:
				good.append(m) # m is from the scene/captured image <DMatch 0x104a34c30>
				#print(m)
		print('# good matches:', len(good))

		return good

	def getMatchLocations(self):
		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) # 3d vectors ? How to get the 3D points??
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) # 2d vectors
			print(np.float32([ kp1[m.queryIdx].pt for m in good ]))
			print(np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2))
			#print(src_pts[0])
			#print(dst_pts[0])

			# Find homography transformation and detect good matches
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()
			h,w,d = roi.image.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)
			img2 = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		else:
			print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None

	def computePose(self, camera_matrix):
		''' find the camera pose for the current frame, by solving PnP problem '''
		pass

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


