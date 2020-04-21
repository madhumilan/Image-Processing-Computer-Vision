import cv2
import numpy as np
import matplotlib.pyplot as plt

MAX_FEATURES = 500
MATCH_PERCENT = 0.15

def alignImages(img1, img2):
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# Detecting ORB (Oriented Fast and Rotated BRIEF) features and descriptors
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
	keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

	# Matching the features
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)
	# print("Matches before sorting", matches)

	# Sort matches by score
	matches.sort(key = lambda x:x.distance, reverse=False)
	# print("After sorting ", matches)

	# Remove not so good matches
	num_goodMatches = round(len(matches) * MATCH_PERCENT)
	print(num_goodMatches)
	matches = matches[:num_goodMatches]

	# Draw the top matches
	matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
	cv2.imwrite("matched.png", matched_img)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find Homography
	homo, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	print(homo)
	height, width, channels = img2.shape
	print(height, width)
	warped = cv2.warpPerspective(img1, homo, (width, height))

	return warped, homo

if __name__ == '__main__':
	image1 = cv2.imread('balea.jpeg')
	# image2 = cv2.rotate(cv2.ROTATE_90_CLOCKWISE, image1)
	image2 = cv2.imread('balea_rotated.jpeg')

	aligned, homo = alignImages(image2, image1)

	cv2.imwrite("aligned.png", aligned)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()