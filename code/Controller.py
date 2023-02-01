from UI import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore 
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np


# from numpy.core.fromnumeric import size
# from scipy import signal

# import sys
# import glob
# import pickle

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.loadImg1Path = None
        self.loadImg2Path = None
        self.loadImg1 = None
        self.loadImg2 = None

    def setup_control(self):
        self.ui.LoadImg1.clicked.connect(self.load_img1_click)
        self.ui.LoadImg2.clicked.connect(self.load_img2_click)
        self.ui.Keypoints.clicked.connect(self.keypoints_click)
        self.ui.MatchedKeypoints.clicked.connect(self.matched_keypoints_click)

    def load_img1_click(self):
        path = os.getcwd()
        self.loadImg1Path  ='./Dataset_CvDl_Hw1_2/Q4_images/box_in_scene.png'
        self.loadImg1= cv2.imread(self.loadImg1Path)
        print(type,self.loadImg1Path)
        
    def load_img2_click(self):
        path = os.getcwd()
        self.loadImg2Path ='./Dataset_CvDl_Hw1_2/Q4_images/box.png'
        self.loadImg2= cv2.imread(self.loadImg2Path)
        print(type,self.loadImg2Path)

    def SIFT(self,img):
        outimg = None
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale-Invariant Feature Transform,SIFT  
        sift = cv2.xfeatures2d.SIFT_create()

        # get the key point
        keypoints , descriptor = sift.detectAndCompute(grayImg, None)

        return img , keypoints , descriptor 


    def keypoints_click(self):
        if self.loadImg1 is None or self.loadImg1Path is None:
            QMessageBox.about(self, "check", "No image,Please confirm the loading image1")
            return
        img = self.loadImg1.copy()
        outImg = None
        img , keypoints , descriptor  = self.SIFT(img)

        # original image , keypoint , output image ,color,flag
        cv2.drawKeypoints(img, keypoints,img, color=(0,255,0))

        cv2.imshow('Keypoints',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def matched_keypoints_click(self):
        if self.loadImg1 is None or self.loadImg1Path is None:
            QMessageBox.about(self, "check", "No image,Please confirm the loading image1")
            return
        if self.loadImg2 is None or self.loadImg2Path is None:
            QMessageBox.about(self, "check", "No image,Please confirm the loading image2")
            return
        img1 = self.loadImg1.copy()
        img2 = self.loadImg2.copy()

        img1 , keypoints1 , descriptor1 = self.SIFT(img1)
        img2 , keypoints2 , descriptor2  = self.SIFT(img2)

        # Match the most related between descriptors 1 and descriptors 2
        bf = cv2.BFMatcher()
        match = bf.knnMatch(descriptor1, descriptor2, k=2)

        #draw the line
        goodMatch = []

        print('len(match)',len(match))
        for m, n in match:
            if m.distance < 0.6 * n.distance:
                goodMatch.append(m)

        goodMatch = np.expand_dims(goodMatch, 1)

        # orginalImg orginalkeypoints , Find Image, Find Image keypoints , Match array , 
        outImg = cv2.drawMatchesKnn(img1, keypoints1,img2,keypoints2,goodMatch,None, matchColor=(127, 127, 255),singlePointColor=(0, 255, 0),flags=0)

        cv2.imshow('Keypoints',outImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
