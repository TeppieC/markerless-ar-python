import cv2
import numpy as np
from matplotlib import pyplot as plt

class Frame(object):
	"""docstring for Marker"""
	def __init__(self, frame):
		self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # a frame
		cv2.imwrite('a.png',frame)
		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()
		# find the keypoints and descriptors with SIFT
		self.keypoints, self.descriptors = sift.detectAndCompute(frame, None)
