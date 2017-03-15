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

class App:
	def __init__(self, cameraMatrix):
		self.referencePoints = []
		self.cropping = False
		self.current_frame = None
		self.cameraMatrix = cameraMatrix

	def chooseMarker(self):
		cap = cv2.VideoCapture(0)

		while True:
 
			# Capture frame-by-frame
			ret, frame = cap.read()

			self.current_frame = frame.copy()
			cv2.namedWindow("frame")
			cv2.setMouseCallback("frame", self.click_and_crop)

			# Our operations on the frame come here
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			# Display the resulting frame
			cv2.imshow('frame',frame)

			# if there are two reference points, then crop the region of interest
			# from teh image and display it
			if len(self.referencePoints) == 2:
				roi = self.current_frame[self.referencePoints[0][1]:self.referencePoints[1][1], \
						self.referencePoints[0][0]:self.referencePoints[1][0]]
				cv2.imshow("ROI", roi)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					cap.release()
					cv2.destroyAllWindows()
					return roi

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break

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
			cv2.rectangle(self.current_frame, self.referencePoints[0], self.referencePoints[1], (0, 255, 0), 2)
			cv2.imshow("image", self.current_frame)

	def processFrames(self):
		cap = cv2.VideoCapture(0)

		while True:
 
			# Capture frame-by-frame
			ret, frame = cap.read()
			self.current_frame = frame.copy()



			cv2.imshow('frame', self.current_frame)
			cv2.imshow('marker', self.roi)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break


	def main(self):
		self.roi = self.chooseMarker()
		self.processFrames()



if __name__ == '__main__':
	app = App()
	app.main()