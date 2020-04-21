import cv2
from cv2 import aruco
import numpy as np

dictionary = aruco.Dictionary_get(aruco.DICT_6X6_1000)

marker = np.zeros((200,200), dtype=np.uint8)
marker = aruco.drawMarker(dictionary, 997, 200, marker, 1)
# pad = np.ones((300,300), dtype=np.uint8)

image = cv2.copyMakeBorder(marker, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255,255,255])

cv2.imshow("Marker with border", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("aruco6X6_1000.png", image)

