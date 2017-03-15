import cv2
import numpy as np
from matplotlib import pyplot as plt

class ROI(object):
	"""docstring for Marker"""
	def __init__(self, image):
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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