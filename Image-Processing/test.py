import cv2
import numpy as np

dhoni = cv2.imread("images/msd_keeping.jpg")
ball = dhoni[440:520, 140:200 ]
r, c, ch = dhoni.shape
print(r, c)
dhoni = dhoni[:, 200:]



cv2.imshow("Dhoni", ball)
cv2.waitKey(0)
cv2.destroyAllWindows()


