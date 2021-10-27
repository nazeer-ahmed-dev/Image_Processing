import cv2
import numpy as np
img =  cv2.imread("image.jpg")
cv2.imshow("Frame",img)
cv2.waitKey(-1)

kernal = np.ones((8,8)).astype(int)
print(kernal)
eros = cv2.erode(img,kernal) #boudnray
cv2.imshow("Frame",eros)
cv2.waitKey(-1)

final = img - eros

second = cv2.erode(eros,kernal)
second1 = cv2.erode(second,kernal)
sc  = second - second1

fm = final + sc
cv2.imshow("Frame",fm)
cv2.waitKey(-1)