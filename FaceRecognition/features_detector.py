import cv2
import numpy as np

img = cv2.imread("E:\\HDA\\ComputerVision\\Python_CV\\FaceRecognition\\face1.jpg", cv2.IMREAD_GRAYSCALE)
print(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img, None)

final = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow("Image with features", final)

cv2.waitKey(0)
cv2.destroyAllWindows()