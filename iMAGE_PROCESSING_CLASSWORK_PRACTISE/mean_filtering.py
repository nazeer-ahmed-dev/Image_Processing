import cv2
import numpy as np
def my_mean(image_gray, size_wind):
    rows_filt, cols_filt = size_wind, size_wind
    row, col = np.shape(image_gray)
    filt_smooth = np.ones((rows_filt, cols_filt)) / (rows_filt*cols_filt)
    # rows_filt, cols_filt = np.shape(filt_smooth)
    pad_rows, pad_cols = np.floor(rows_filt / 2), np.floor(cols_filt / 2)
    image_output = np.zeros((row, col))
    image_padded = np.pad(image_gray, (int(pad_rows), int(pad_cols)), 'symmetric')
    for i in range(row):
        for j in range(col):
            patch_curr = image_padded[i:i + rows_filt, j:j + cols_filt]
            array_mult = patch_curr * filt_smooth
            image_output[i, j] = np.sum(array_mult)

    cv2.imshow("f",image_output)
    print("F")
    cv2.waitKey(-1)

im = cv2.imread('noise.tif')
img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
my_mean(img,3)