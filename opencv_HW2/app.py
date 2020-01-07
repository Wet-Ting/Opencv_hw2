############################################################################
#                              Cv.hw1                                      #  
#                        Arthor: Wet-ting Cao.                             #   
#                             2019.10.24                                   #
############################################################################

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsView, QGraphicsScene
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from mainwindow import Ui_MainWindow
import cv2 as cv
import numpy as np  

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math

global pt
pt = []

# MainWindow -> button implementation.
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        # Q1.
        self.btn1_1.clicked.connect(self.disp)
        
        # Q2.
        self.btn2_1.clicked.connect(self.bs)
        
        # Q3.
        self.btn3_1.clicked.connect(self.pre)
        self.btn3_2.clicked.connect(self.vt)
        
        # Q4.
        self.btn4_1.clicked.connect(self.ar)
        
    def disp(self):
        imgL = cv.imread('imL.png', 0)
        imgR = cv.imread('imR.png', 0)

        stereoL = cv.StereoSGBM_create(numDisparities = 64, blockSize = 9, P1 = 1400, P2 = 2000)        
        disL = stereoL.compute(imgL, imgR)

        plt.figure(), plt.title('Without L-R Disparity Check'), plt.axis('off'), plt.imshow(disL, 'gray')
        plt.show()
        
    def bs(self):
        cap = cv.VideoCapture('bgSub.mp4')
        fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history = 50)
        
        while True:
            r, f = cap.read()
            
            if r != True:
                break
                
            fgmask = fgbg.apply(f)           
            cv.imshow('test1', f)
            cv.imshow('test2', fgmask)
            
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv.destroyAllWindows()
            
    def pre(self):
        cap = cv.VideoCapture('featureTracking.mp4')
        
        r, f = cap.read()               
        f = cv.convertScaleAbs(f)
        params = cv.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.85
        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(f)
        
        for i in range(len(keypoints)):
            if i == 6:
                continue
            img = cv.rectangle(f, (round(keypoints[i].pt[0] - 5.5), round(keypoints[i].pt[1] - 5.5)), (round(keypoints[i].pt[0] + 5.5), round(keypoints[i].pt[1] + 5.5)) ,(0, 0, 255), 1)
        cv.imshow('frame', img)
        
    def vt(self):
        cap = cv.VideoCapture('featureTracking.mp4')
        color = np.random.randint(0, 255, (100, 3))
        
        r, f = cap.read()   
        old_gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        f = cv.convertScaleAbs(f)
        params = cv.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.85
        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(f)  
        mask = np.zeros_like(f)
               
        p0 = np.array([[[np.float32(keypoints[0].pt[0]), np.float32(keypoints[0].pt[1])]], 
                [[np.float32(keypoints[1].pt[0]), np.float32(keypoints[1].pt[1])]],
                [[np.float32(keypoints[2].pt[0]), np.float32(keypoints[2].pt[1])]],
                [[np.float32(keypoints[3].pt[0]), np.float32(keypoints[3].pt[1])]],
                [[np.float32(keypoints[4].pt[0]), np.float32(keypoints[4].pt[1])]],
                [[np.float32(keypoints[5].pt[0]), np.float32(keypoints[5].pt[1])]],
                [[np.float32(keypoints[7].pt[0]), np.float32(keypoints[7].pt[1])]]])

        while True:
            ret, frame = cap.read()
            if ret != True:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                  
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize = (21, 21), maxLevel = 5, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.04))
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
            
            img = cv.add(frame, mask)
            cv.imshow('frame', img)
            
            k = cv.waitKey(30) #& 0xff
            if k == 27:
                break
                
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
        cv.destroyAllWindows()
        cap.release()
    
    def ar(self):
        print('Augmented Reality')
        def draw(img, corner, imgpt) :
            imgpt = np.int32(imgpt).reshape(-1, 2)
            img = cv.drawContours(img, [imgpt[:4]], -1, (0, 0, 255), 3)
            for i in range(4) :
                img = cv.line(img, tuple(imgpt[i]), tuple(imgpt[i + 4]), (0, 0, 255), 3)
                
            return img
            
        images = []
        objp = []
        imgp = []
        path = 'ar/'
        obj = np.zeros((11 * 8, 3), np.float32)
        obj[:, : 2] = np.mgrid[0 : 11, 0 : 8].T.reshape(-1, 2)
        criter = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        axis = np.float32([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0],
                   [3, 3, -4], [3, 3, -4], [3, 3, -4], [3, 3, -4]])
        
        for i in range(5) :
            img = cv.imread(path + str(i + 1) + '.bmp')
            images.append(img)
        
        for img in images:
            to_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corner = cv.findChessboardCorners(to_gray, (11, 8), None)
            
            if ret == True :
                objp.append(obj)
                corners = cv.cornerSubPix(to_gray, corner, (11, 11), (-1, -1), criter)
                imgp.append(corners)
                
                img = cv.drawChessboardCorners(img, (11, 8), corner, ret)
        
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', 512, 512)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objp, imgp, to_gray.shape[::-1], None, None)
        
        images.clear()
        for i in range(5) :
            img = cv.imread(path + str(i + 1) + '.bmp')
            images.append(img)
        
        for img in images:
            to_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corner = cv.findChessboardCorners(to_gray, (11, 8), None)
            if ret == True :
                corners = cv.cornerSubPix(to_gray, corner, (11, 11), (-1, -1), criter)
                _, rvecs, tvecs, inliers = cv.solvePnPRansac(obj, corners, mtx, dist)
                imgpt, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = draw(img, corners, imgpt)
                cv.imshow('Image', img)
                cv.waitKey(500)
                
        cv.destroyAllWindows()
        
                              
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())