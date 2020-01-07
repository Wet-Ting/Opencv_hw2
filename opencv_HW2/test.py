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

imgL = cv.imread('imL.png', 0)
imgR = cv.imread('imR.png', 0)

stereoL = cv.StereoSGBM_create(numDisparities = 64, blockSize = 9, P1 = 1400, P2 = 2000)        
disL = stereoL.compute(imgL, imgR)

plt.figure(), plt.title('Without L-R Disparity Check'), plt.axis('off'), plt.imshow(disL, 'gray')
plt.show()