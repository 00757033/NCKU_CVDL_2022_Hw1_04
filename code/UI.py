# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(384, 281)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 20, 191, 201))
        self.groupBox.setObjectName("groupBox")
        self.LoadImg1 = QtWidgets.QPushButton(self.groupBox)
        self.LoadImg1.setGeometry(QtCore.QRect(20, 30, 121, 23))
        self.LoadImg1.setObjectName("LoadImg1")
        self.LoadImg2 = QtWidgets.QPushButton(self.groupBox)
        self.LoadImg2.setGeometry(QtCore.QRect(20, 70, 121, 23))
        self.LoadImg2.setObjectName("LoadImg2")
        self.Keypoints = QtWidgets.QPushButton(self.groupBox)
        self.Keypoints.setGeometry(QtCore.QRect(20, 110, 121, 23))
        self.Keypoints.setObjectName("Keypoints")
        self.MatchedKeypoints = QtWidgets.QPushButton(self.groupBox)
        self.MatchedKeypoints.setGeometry(QtCore.QRect(20, 150, 121, 23))
        self.MatchedKeypoints.setObjectName("MatchedKeypoints")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 384, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "4.SIFT"))
        self.LoadImg1.setText(_translate("MainWindow", "Load Image 1"))
        self.LoadImg2.setText(_translate("MainWindow", "Load Image 2"))
        self.Keypoints.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.MatchedKeypoints.setText(_translate("MainWindow", "4.2 Matched Keypoints"))


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())
