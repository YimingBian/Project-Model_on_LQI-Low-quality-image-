import cv2 as cv
from cv2 import blur
import numpy as np

objects = ["dog", "crayfish", "bike42"]

for object in objects:
    im1 = cv.imread("./original/{0}.jpg".format(object))
    im1_wid = im1.shape[0]
    im1_hei = im1.shape[1]    
    # blur
    blur_kernel_size = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in blur_kernel_size:
        tmp = cv.blur(im1, (i,i))
        cv.imwrite('./blur/{0}_blur_{1}.jpg'.format(object, i), tmp)
    
    # crop
    proportions = [90, 80, 70, 60, 50]
    for p in proportions:
        p_adj = (100-p)/200
        wid_low = int(im1_wid * p_adj)
        wid_high = int(im1_wid - wid_low)
        hei_low = int(im1_hei * p_adj)
        hei_high = int(im1_hei - hei_low)
        tmp = im1[wid_low:wid_high, hei_low:hei_high]
        cv.imwrite('./crop/{0}_crop_{1}.jpg'.format(object, p), tmp)

    # grayscale and binary
    tmp1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    tmp1_pad = np.stack((tmp1,)*3, axis=-1)
    cv.imwrite('./color/{0}_gray.jpg'.format(object), tmp1_pad)	
    (thresh, tmp2) = cv.threshold(tmp1, 127, 255, cv.THRESH_BINARY)
    tmp2_pad = np.stack((tmp2,)*3, axis=-1)
    cv.imwrite('./color/{0}_bw.jpg'.format(object), tmp2_pad)
