import cv2 as cv

from userlib import *
imgconvtogray('images','grayimages')
imgresize('images','resizedimages')
imgAugment('images','augmented')
imgEdgeDetect('images','edgedimages')
imgthresholdglobal('images','thresholdgloabl')
imgthresholdgaussian('images','thresholdadaptivegaussian')
imghisequalizer('images','histEqualimages')