import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def imgClahe(img):
    if img.ndim != 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5)
    img = clahe.apply(img)
    return img

def imgNormalEqualizer(img):
    if img.ndim != 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    return img

def imgshow(rootfolder):
    imagefolders = os.listdir(rootfolder)
    noofclass = 0

    for folder in imagefolders:
        loc = rootfolder + "/" + folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)

        for image in files:
            location = rootfolder + '/' + folder + '/' + image
            img = cv.imread(location)
            watchTimeinSeconds = .3
            imgview(img, watchTimeinSeconds)
    print("Number of class: ", noofclass)

def augmentation(img):
    def rotate_image(image):
        rotated = cv.rotate(image, cv.ROTATE_180)
        return rotated

    def flip_image(image, flip_code=1):
        return cv.flip(image, 1)

    def add_noise(image):
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return rotate_image(img), flip_image(img), add_noise(img)


def __imgThresholdApadtiveMean(img):
    if img.ndim != 2:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    adaptive_thresh_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3)
    return adaptive_thresh_mean


def __imgThresholdApadtiveGauss(img):
    if img.ndim != 2:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    adaptive_thresh_gauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 3)
    return adaptive_thresh_gauss

def test(img):
    return __imgThresholdApadtiveGauss(img)

def __imgThresholdGlobal(img):
    if img.ndim != 2:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    threshold, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    return thresh


def imgview(img, watchTime):
    # img=imgresize(img)
    cv.imshow('window', img)
    delay = int(watchTime * 1000)
    cv.waitKey(delay)


def __imghist(img):
    if len(img.shape) == 2:
        # Grayscale image
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.figure()
        plt.title('Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of Pixels')
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    else:
        #Color image
        colors = ('b', 'g', 'r')
        plt.figure()
        plt.title('Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of Pixels')
        for i, col in enumerate(colors):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()


def imgblur(src, des):
    imagefolders = os.listdir(src)
    noofclass = 0

    for folder in imagefolders:
        loc = src + "/" + folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff = True
        for image in files:
            location = src + '/' + folder + '/' + image
            img = cv.imread(location)
            img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
            destination = des + '/' + folder + '/' + image

            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des + '/' + folder)
            except:
                pass
            cv.imwrite(destination, img)


def imgresize(src, des):
    imagefolders = os.listdir(src)
    noofclass = 0
    for folder in imagefolders:
        loc = src + "/" + folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff = True
        for image in files:
            location = src + '/' + folder + '/' + image
            img = cv.imread(location)
            img = cv.resize(img, (512, 512))
            destination = des + '/' + folder + '/' + image
            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des + '/' + folder)
            except:
                pass
            cv.imwrite(destination, img)
def imgthresholdglobal(src,des):
    imagefolders = os.listdir(src)
    noofclass=0
    for folder in imagefolders:
        loc=src+"/"+folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff=True
        for image in files:
            location = src+'/'+folder+'/'+image
            img=cv.imread(location)
            img=__imgThresholdGlobal(img)
            destination = des+'/'+folder+'/'+image
            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des+'/'+folder)
            except:
                pass
            cv.imwrite(destination,img)
def imgthresholdgaussian(src,des):
    imagefolders = os.listdir(src)
    noofclass=0
    for folder in imagefolders:
        loc=src+"/"+folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff=True
        for image in files:
            location = src+'/'+folder+'/'+image
            img=cv.imread(location)
            img=__imgThresholdApadtiveGauss(img)
            destination = des+'/'+folder+'/'+image
            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des+'/'+folder)
            except:
                pass
            cv.imwrite(destination,img)
def imgthresholdmean(src,des):
    imagefolders = os.listdir(src)
    noofclass=0
    for folder in imagefolders:
        loc=src+"/"+folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff=True
        for image in files:
            location = src+'/'+folder+'/'+image
            img=cv.imread(location)
            img=__imgThresholdApadtiveMean(img)
            destination = des+'/'+folder+'/'+image
            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des+'/'+folder)
            except:
                pass
            cv.imwrite(destination,img)
def imgconvtoGray(src, des):
    imagefolders = os.listdir(src)
    noofclass = 0
    for folder in imagefolders:
        loc = src + "/" + folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff = True
        for image in files:
            location = src + '/' + folder + '/' + image
            img = cv.imread(location)
            img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
            destination = des + '/' + folder + '/' + image
            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des + '/' + folder)
            except:
                pass
            cv.imwrite(destination, img)


def imgEdgeDetect(src, des):
    imagefolders = os.listdir(src)
    noofclass = 0
    for folder in imagefolders:
        loc = src + "/" + folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)
        ff = True
        for image in files:
            location = src + '/' + folder + '/' + image
            img = cv.imread(location)
            img = cv.Canny(img, 125, 175)
            destination = des + '/' + folder + '/' + image
            try:
                os.mkdir(des)
            except:
                pass
            try:
                os.mkdir(des + '/' + folder)
            except:
                pass
            cv.imwrite(destination, img)


def imgshowhist(rootfolder):
    imagefolders = os.listdir(rootfolder)
    noofclass = 0

    for folder in imagefolders:
        loc = rootfolder + "/" + folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)

        for image in files:
            location = rootfolder + '/' + folder + '/' + image
            img = cv.imread(location)
            __imghist(img)
    print("Number of class: ", noofclass)
