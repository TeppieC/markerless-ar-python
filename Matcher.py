'''
Author: Zhaorui Chen 2017

Code based on OpenCV doc - python tutorial.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 50

class Matcher:
	def __init__(self, roi, alg, distCoeffs, cameraMatrix):
		self.roi = roi # a ROI pattern object
		self.alg = alg
		self.cameraMatrix = cameraMatrix
		self.distCoeffs = distCoeffs

	def setFrame(self, frame):
		self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	def getCorrespondence(self):
		# Get the previously extracted keypoints and descriptors for the marker object
		kp1 = self.roi.keypoints
		des1 = self.roi.descriptors

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


		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		# simple nn, can be refined with cross validation
		# des1: descriptors from the marker
		# des2: descriptors from the frame
		matches = flann.knnMatch(des1,des2,k=2) # [ ..., [<DMatch 0x104a34c30>, <DMatch 0x104a34c50>], ... ]

		# store all the good matches as per Lowe's ratio test.
		good = [] # good is a container for 3D points found in scene
		for m_n in matches:
			if len(m_n)!=2:
				continue
			(m, n) = m_n
			if m.distance < 0.7*n.distance:
				good.append(m) # m is from the scene/captured image <DMatch 0x104a34c30>


		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) # n*(1*2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) # n*(1*2)

			# Find homography transformation and detect good matches, using the matching image point pairs
			# bring points from a pattern to the query image coordinate system
			# see Homography Transformation
			# API: Finds a perspective transformation between two planes.
			# http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findhomography
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # M is the transformation matrix, thresholding at 5
			matchesMask = mask.ravel().tolist()

			h,w = self.roi.image.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) # 4 corners of the roi pattern image

			try:
				# find the coordinates of the corners of the marker, in the frame image coordination
				# http://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#void%20perspectiveTransform(InputArray%20src,%20OutputArray%20dst,%20InputArray%20m)
				corners = cv2.perspectiveTransform(pts,M) # points2d for the frame image
			except:
				print('No matching points after homography estimation')
				return

			print('enough,',len(good))
			return (src_pts, dst_pts, corners)

		else:
			print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None
			return None

	def computePose(self, src, dst):
		''' find the camera pose for the current frame, by solving PnP problem '''

		'''
		cv2.solvePnP(objectPoints, imagePoints, 
			cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) 
		→ retval, rvec, tvec
		'''
		retval, rvec, tvec = cv2.solvePnP(src, dst, self.cameraMatrix, self.distCoeffs)
		return (retval, rvec, tvec)

