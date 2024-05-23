import cv2
import cv2 as cv
import os
def imgresize(img):
    img2 = img
    while(img2.shape[0]>512):
        img2=cv.resize(img2,(0,0),fx=.9,fy=.9)
    return img2
def imgshow(rootfolder):
    imagefolders = os.listdir(rootfolder)
    noofclass=0

    for folder in imagefolders:
        loc=rootfolder+"/"+folder
        noofclass = len(imagefolders)
        files = os.listdir(loc)

        for image in files:
            location = rootfolder+'/'+folder+'/'+image
            img=cv.imread(location)
            watchTimeinSeconds=.3
            imgview(img,watchTimeinSeconds)
    print("Number of class: ",noofclass)
def imgview(img,watchTime):
    # img=imgresize(img)
    cv.imshow('window',img)
    delay = int(watchTime*1000)
    cv.waitKey(delay)


def imgblur(src,des):
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
            img=cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
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
def imgresize(src,des):
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
            img=cv.resize(img,(512,512))
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
def imgconvtoGray(src,des):
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
            img=cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
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
            img = cv.Canny(img, 125,175)
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


imgshow('images')
imgresize('images','pro1')
imgEdgeDetect('pro1','pro1')
imgshow('pro1')
# imgresize('images','processed')
# imgblur('processed','processed')
# imgconvtoGray('processed','processed')
#
# imgshow('processed')





cv.destroyAllWindows()
