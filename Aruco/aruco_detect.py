import cv2
import numpy as np

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)

#marker = np.zeros((200,200), dtype=np.uint8)
#marker = cv2.aruco.drawMarker(dictionary, 33, 200, marker, 1)

#cv2.imwrite("marker.png", marker)

# Creating the Aruco Detection params ()
params = cv2.aruco.DetectorParameters_create()
print(params)

frame = cv2.imread("aruco6X6_1000.png")

markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)
print(markerCorners)
print(markerIds)
print(rejectedCandidates)
# cv2.rectangle(frame, ((markerCorners[0][0],markerCorners[0][1]), (markerCorners[1][0],markerCorners[1][1]), 
# 	(markerCorners[2][0],markerCorners[2][1]), (markerCorners[3][0], markerCorners[3][1])), (0, 255, 0), 2)

cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

cv2.imshow("Frame", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()