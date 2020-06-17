import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

try:
	img = cv2.imread("checkerboard.png")
except:
	print("Image not found.")

print img

width = int(img.shape[0] * 0.3)
height = int(img.shape[1] * 0.55)

resized = (cv2.resize(img, (width,height), cv2.INTER_AREA))
print(resized.shape)
cv2.imwrite("images/resized_dettol.jpg", resized)

print("Blue value = ",img[100,100, 0])			# Accessing individual pixel as well as it's color channel

Using numpy method to get the pixel value
print(img.item(100,100,0))
img.itemset((100,100,0),0)			# Setting a pixel value for particular channel
print(img.item(100,100,0))

Image Properties
print(resized.shape)
print(img.size)
print(img.dtype)

Image ROI
roi = resized[70:180, 100:600]

Drawing border around an image
bordered1 = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0,255,125))
bordered2 = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
cv2.imwrite("images/bordered.jpg", bordered1)
plt.subplot(211), plt.imshow(bordered1,"gray"), plt.title("COnstant Border")
plt.subplot(212), plt.imshow(bordered2,"gray"), plt.title("Border Replicate")
plt.show()			# Blue and Red colors vary as matplotlib used RGB format

Arithmetic Operations
x = np.uint8([250])
y = np.uint8([15])
print(cv2.add(x,y))
print(x+y)
copy1 = resized.copy()
copy2 = resized.copy()
# sum = copy1+copy2
sum = cv2.add(copy1, copy2)

Image transparency using cv2.addWighted()
img1 = cv2.imread("images/test_thumbnail.jpg")
img1 = cv2.resize(img1, (480,660))
img2 = cv2.imread("images/resized_dettol.jpg")
blended = cv2.addWeighted(img1, 0.1, img2, 0.5, 0)


# Bitwise Operations (Embedding one image onto another)
t1 = cv2.getTickCount()
time1 = time.time()

img1 = cv2.imread("images/opencv_logo.png")
logo = cv2.resize(img1, (300, 400))
img2 = cv2.imread("images/msd.jpg")

# ROI
row, col, channels = logo.shape
roi = img2[0:row, 0:col]
# roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Creation of mask
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# TO black out the mask region in main image
img2_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

# Take only region of image from the logo
img1_fg = cv2.bitwise_and(logo, logo, mask = mask)

joined = cv2.add(img1_fg, img2_bg)
img2[:row, :col] = joined

t2 = cv2.getTickCount()
time2 = time.time()
time_taken = (t2 - t1) / cv2.getTickFrequency()
print("Total time =", (time2-time1))
print("Time taken =", time_taken, "secs")


# Translation
M = np.float32([[1, 0, 100], [0, 1, 50]])			# Move 100 pos in x and 50 pos in y
height, width = resized.shape[:2]
dst = cv2.warpAffine(resized, M, (width, height))

# Rotation
M = cv2.getRotationMatrix2D((width/2,height/2), 90, 1)
dst = cv2.warpAffine(resized, M, (width, height))
print(dst.shape[:2])

# ----------------------------------------------------------
# Color spaces in Opencv
resized_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)

cs_flags = [cs for cs in dir(cv2) if cs.startswith("COLOR_")]
print(cs_flags)
print(cs_flags.count)


# Object tracking using HSV color space
cap = cv2.VideoCapture(0)

while 1:
	# Take each frame
	_, frame = cap.read()

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	print(hsv[250,350])
	# Define range of blue in the frame
	low_blue = np.array([105,55,55])
	high_blue = np.array([135,255,255])
	low_orange = np.array([5,50,50])
	high_orange = np.array([15,255,255])

	# # To find the HSV value of green
	# green_bgr = np.uint8([[[0,255,0]]])
	# hsv_green = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2HSV)
	# print(hsv_green)

	# Threshold the image frame to include only the blue objects
	# mask1 = cv2.inRange(hsv, low_orange, high_orange)
	mask2 = cv2.inRange(hsv, low_blue, high_blue)

	# AND with the mask
	# res_orange = cv2.bitwise_and(frame, frame, mask=mask1)
	res_blue = cv2.bitwise_and(frame, frame, mask=mask2)

	cv2.imshow("Frame", frame)
	cv2.imshow("Mask Orange", mask2)
	# cv2.imshow("Mask Blue", mask2)
	cv2.imshow("Blue object tracking", res_blue)

	key = cv2.waitKey(5) & 0xff
	if key == 27:
		break
cv2.destroyAllWindows()

# -----------------------------------------------------------------
# Image Thresholding
gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Adaptive thresholding
img1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
img2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# OTSU's thresholding
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Otsu's threshold =", ret)

cv2.imshow("Binary Threshold", img)
cv2.imshow("Adaptive Mean", img1)
cv2.imshow("Adaptive Gaussian", img2)
cv2.imshow("OTSU", otsu)

# -------------------------------------------
# Affine transformation
pts1 = np.float32([[10,20],[280,300],[600,350]])
pts2 = np.float32([[50,60],[300,350],[550,400]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(resized, M, (480,660))

cv2.imshow("Original", resized)
cv2.imshow("Warped", dst)

# --------------------------------------------
# Perspective Transform
sudoku = cv2.imread("images/sudoku.png")
plt.subplot(121), plt.imshow(sudoku)

pts1 = np.float32([[48,53],[350,52],[18,365],[370,370]])
pts2 = np.float32([[0,0],[350,0],[0,350],[350,350]])
M = cv2.getPerspectiveTransform(pts1, pts2)

res = cv2.warpPerspective(sudoku, M, (350,350))
plt.subplot(122), plt.imshow(res), plt.title("Result"), plt.show()

# ---------------------------------------------
# Averaging filters using kernels
kernel = np.ones((5,5), dtype=np.float32)/25
dst = cv2.filter2D(resized, -1, kernel)

plt.subplot(211), plt.imshow(resized)
plt.subplot(212), plt.imshow(dst)
plt.show()

# ---------------------------------------------
# Morphological Processing
coins = cv2.imread("images/coins1.png", 0)
# ret, binary = cv2.threshold(coins, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, binary = cv2.threshold(coins, 35, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
kernel = np.ones((9,9), np.uint8)
dilate = cv2.dilate(eroded, kernel, iterations=1)

# ----------------------------------------------
# Edge detection
sudoku = cv2.imread("images/sudoku.png", 0)
sobel_x = cv2.Sobel(sudoku, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow("Eroded", sobel_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

Canny Edge Detection
sudoku = cv2.imread("images/sudoku.png", 0)
canny = cv2.Canny(sudoku, 75, 150)

plt.subplot(121), plt.imshow(sudoku, cmap='gray'), plt.title("Original")
plt.subplot(122), plt.imshow(canny, cmap='gray'), plt.title("Canny Edge image")
plt.show()

x = list(range(10))
print("x =",x)
pos = x[5:0:-1]
print("pos =",pos)
neg = x[5:0:-1]
print("neg =",neg)

# --------------------------------------------------------------------------
# Harris Corner Detector
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
print "Image size =", gray.shape
cornersImage = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
print "Max =", cornersImage.max()
print "size =", cornersImage.shape
img[cornersImage>0.01*(cornersImage.max())] = [0,0,255]

cv2.imshow('Corner', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------------------
Corners with subpixel accuracy - cv2.cornerSubPix()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cornersImage = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(cornersImage, None)
ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
cv2.imshow("Dilated", dst)
dst = np.uint8(dst)

#Find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
print centroids

#Define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

#Now draw the corners
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
cv2.imshow("Subpixel corners", img)

# --------------------------------------------------------------------------
SIFT - Feature extraction /[Need to be installed]/
img = cv2.imread('cone.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp)
cv2.imshow("Key points", img)

Fast feature detector
img = cv2.imread('cone.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image",img)
fast = cv2.FastFeatureDetector()
print "Fast detector created."
print fast
keypoints = fast.detect(img, None)			#Segmentation fault
print "keypoints detected."
img2 = cv2.drawKeypoints(img, keypoints, color=(0,255,0))
cv2.imshow("Features", img2)

#ORB feature detector and matching
orb = cv2.ORB()
keypoints = orb.detect(img, None)
keypoints, descriptors = orb.compute(img, keypoints)

result = cv2.drawKeypoints(img, keypoints, color=(0,255,0), flags=0)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()