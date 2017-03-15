'''
SIZE = 1280, 720
Distortion factor: k1=-0.0481117368, k2=0.1561547816, p1=0.0549917854, p2=-0.0129148327
				  fx=1007.356812, fy=1033.845215, x0=642.705933, y0=517.837097, s=0.854053
1179.50112 0.00000 642.70593 0.00000 
0.00000 1210.51605 517.83710 0.00000 
0.00000 0.00000 1.00000 0.00000 
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
from ROI import ROI
from Matcher import Matcher
from Frame import Frame

disCoeff = np.float32([0.0481117368, 0.1561547816, 0.0549917854, -0.0129148327])
cameraMatrix = np.float32([[1007.356812, 0.0, 642.705933], [0.0, 1033.845215, 517.837097],[0.0,0.0,1.0]])

class App:
	def __init__(self, alg):
		self.referencePoints = []
		self.cropping = False
		#self.cameraMatrix = cameraMatrix
		self.roi = None
		self.alg = alg

	def click_and_crop(self, event, x, y, flags, param):
	 
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			self.referencePoints = [(x, y)]
			self.cropping = True
	 
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			self.referencePoints.append((x, y))
			self.cropping = False
	 
			# draw a rectangle around the region of interest
			#cv2.rectangle(self.currentFrame, self.referencePoints[0], self.referencePoints[1], (0, 255, 0), 2)
			#cv2.imshow("frame", self.currentFrame)

	def draw(self, img, corners, imgpts):
	    corner = tuple(corners[0].ravel())
	    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
	    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
	    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
	    return img

	def main(self):
		# crop the webcam frame to get the marker pattern
		cap = cv2.VideoCapture(0)

		while True:
 
			# Capture frame-by-frame
			ret, frame = cap.read()

			currentFrame = frame.copy()
			cv2.namedWindow("choose marker")
			cv2.setMouseCallback("choose marker", self.click_and_crop)

			# Our operations on the frame come here
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			# Display the resulting frame
			cv2.imshow('choose marker',frame)

			# if there are two reference points, then crop the region of interest
			# from teh image and display it
			if len(self.referencePoints) == 2:
				cropImage = currentFrame[self.referencePoints[0][1]:self.referencePoints[1][1], \
						self.referencePoints[0][0]:self.referencePoints[1][0]]
				#cap.release()
				cv2.rectangle(currentFrame, self.referencePoints[0], self.referencePoints[1], (0, 255, 0), 2)
				#cv2.destroyAllWindows()
				# initialize a pattern object for the marker
				self.roi = ROI(cropImage, self.alg)

				cv2.imshow('choose marker',currentFrame)
				cv2.waitKey(1000)
				#cap.release()				
				cv2.destroyWindow('choose marker')
				break
				#if cv2.waitKey(1) & 0xFF == ord('q'):
				#	cap.release()
				#	cv2.destroyAllWindows()
				#	return roi

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break


		
		# handling the logic for pattern matching ...
		#cap2 = cv2.VideoCapture(0)
		cv2.waitKey(100)
		matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)

		while True:
			# Capture frame-by-frame
			ret, frame = cap.read()
			currentFrame = frame.copy()

			cv2.namedWindow('webcam')
			cv2.imshow('webcam', currentFrame)
			cv2.waitKey(100)

			matcher.setFrame(currentFrame)
			matcher.getCorrespondence()
			# TODO: need to show the correspondences

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break
		


if __name__ == '__main__':
	app = App('sift')
	app.main()