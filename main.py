'''
OLD
SIZE = 1280, 720
Distortion factor: k1=-0.0481117368, k2=0.1561547816, p1=0.0549917854, p2=-0.0129148327
				  fx=1007.356812, fy=1033.845215, x0=642.705933, y0=517.837097, s=0.854053
1179.50112 0.00000 642.70593 0.00000 
0.00000 1210.51605 517.83710 0.00000 
0.00000 0.00000 1.00000 0.00000 


'''
#SIZE = 1280, 720
k1=0.0814318880
k2=-0.0729859173
p1=0.0077049430
p2=-0.0046416679
fx=1019.955688
fy=1019.131287
x0=643.416321
y0=381.590668
s=0.957716
'''
1064.98771 0.00000 643.41632 0.00000 
0.00000 1064.12691 381.59067 0.00000 
0.00000 0.00000 1.00000 0.00000 
---------------------------------
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
from ROI import ROI
from Matcher import Matcher
import argparse

disCoeff = np.float32([k1, k2, p1, p2])
cameraMatrix = np.float32([[fx, 0.0, x0], [0.0, fy, y0],[0.0,0.0,1.0]])

class App:
	def __init__(self, alg, mode):
		self.referencePoints = []
		self.cropping = False
		#self.cameraMatrix = cameraMatrix
		self.roi = None
		self.alg = alg
		self.mode = mode

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

		if self.mode == 'capture':
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
					#cv2.imwrite('roi.png',cropImage)


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
		else:
			roi = cv2.imread('roi.png')
			self.roi = ROI(roi, self.alg)


		
		# handling the logic for pattern matching 
		#cap2 = cv2.VideoCapture(0)
		cv2.waitKey(100)
		matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)

		while True:
			# Capture frame-by-frame
			ret, frame = cap.read()
			currentFrame = frame.copy()
			mirrorFrame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
		
			'''
			plt.subplot(2,1,1),plt.imshow(self.roi.image)
			plt.subplot(2,1,2),plt.imshow(currentFrame)
			plt.show()
			'''

			cv2.namedWindow('webcam')
			cv2.imshow('webcam', currentFrame)
			
			matcher.setFrame(currentFrame)
			

			result = matcher.getCorrespondence()
			if result:
				# get the corners
				(src, dst, corners) = result
			else:
				# Not enough matching points found
				print('Not enough points')
				cv2.waitKey(1)
				continue

			#(src, dst, corners) = matcher.getCorrespondence()
			print('enough points')

			#(retval, rvec, tvec) = matcher.computePose(src, dst)
			(retvalCorner, rvecCorner, tvecCorner) = matcher.computePose(self.roi.getPoints3dnew(), corners)
			if retvalCorner:
				# find where to draw the axises of the the cube in the frame        
				#width, height = self.roi.image.shape
				#axis = np.float32([[0,0,0],[0,width,0],[width,height,0],[0,0,height]])

				axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

				'''
				cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs
					[, imagePoints[, jacobian[, aspectRatio]]]) 
				→ imagePoints, jacobian
				'''
				# project 3d points to 2d coordinates in the frame coordination
				imgpts, jac = cv2.projectPoints(axis, rvecCorner, tvecCorner, cameraMatrix, disCoeff)
				#imgpts2, jac2 = cv2.projectPoints(axis, rvecCorner, tvecCorner, cameraMatrix, disCoeff)

				print('corners',[corners])
				print('imgpts[:4]',[imgpts[:4]])

				# re-draw the frame
				currentFrame = cv2.polylines(currentFrame,[np.int32(corners)],True,255,3, cv2.LINE_AA)
				currentFrame = self.renderImg(currentFrame, imgpts, mirrorFrame)

				cv2.imshow('webcam', currentFrame)
				#plt.subplot(2,1,2),plt.imshow(currentFrame)
				#plt.show()
				cv2.waitKey(1)
				
				#cv2.waitKey(1)
				# TODO: need to show the correspondences
			else:
				#print('not able to solve pnp')
				cv2.waitKey(1)
				continue

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break

	def renderImg(self, img, imgpts, mirrorFrame):
		imgpts = np.int32(imgpts).reshape(-1,2)
		# draw ground floor in green

		img = cv2.drawContours(img, [imgpts[:4]],-1,(175,175,175),-3)

		# draw pillars in blue color
		for i,j in zip(range(4),range(4,8)):
			img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

		x_offset=y_offset=50
		x_offset = imgpts[0][0]
		#print(x_offset)
		y_offset = imgpts[0][1]
		img[y_offset:y_offset+mirrorFrame.shape[0], x_offset:x_offset+mirrorFrame.shape[1]] = mirrorFrame

		'''
		# draw top layer in red color
		img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
		'''
		return img


if __name__ == '__main__':
	app = App('sift','static')
	#app = App('sift', 'capture')
	app.main()