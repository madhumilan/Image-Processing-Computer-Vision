import cv2
import numpy as np

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)

marker = np.zeros((200,200), dtype=np.uint8)
marker = cv2.aruco.drawMarker(dictionary, 997, 200, marker, 1)

cv2.imwrite("marker6_1000.png", marker)