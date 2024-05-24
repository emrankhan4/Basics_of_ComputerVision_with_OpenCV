import cv2 as cv

from userlib import *
# imgshowhist('images')

# imgshow('images')
# imgresize('images','pro1')
# imgEdgeDetect('pro1','pro1')
# imgshow('pro1')

# imgresize('images','processed')
# imgblur('processed','processed')
# imgconvtoGray('processed','processed')
#
# imgshow('processed')
# imgconvtoGray('images','grayimg')

# imgshowhist('grayimg')

# cv.destroyAllWindows()
# img1 = cv.imread('med.png')
# cv.equalizeHist(img1,img1)
# cv.imshow(img1)
# cv.waitKey(0)
imgthresholdgaussian('images','gausthres')