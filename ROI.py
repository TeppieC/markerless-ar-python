import cv2
import numpy as np
from matplotlib import pyplot as plt

class ROI(object):
	"""docstring for Marker"""
	def __init__(self, image, alg):
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if alg == 'orb':
			# Initiate ORB detector
			orb = cv2.ORB_create()
			# find the keypoints and descriptors with ORB
			self.keypoints, self.descriptors = orb.detectAndCompute(self.image, None)
		elif alg == 'sift':
			# Initiate SIFT detector
			sift = cv2.xfeatures2d.SIFT_create()
			# find the keypoints and descriptors with SIFT
			self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)
			
		width, height = self.image.shape

		# normalize
		maxSize = max(width, height)
		w = width/maxSize
		h = height/maxSize

		self.points2d = np.array([[0,0],[width,0],[width,height],[0,height]])
		self.points3d = np.array([[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0]])

	def getPoints2d(self):
		return self.points2d

	def getPoints3d(self):
		return self.points3d