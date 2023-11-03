import sys
import typing
import threading
from PyQt5 import QtCore, QtGui
import copy
import cv2

from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit, QWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from UI.Ui_face_recognition_gui import Ui_MainWindow


class multithread_UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cap = cv2.VideoCapture()
        self.src_image = None

        self.stopEvent = threading.Event()  # 默认是False
        self.lock = threading.Lock()

        self.background()

    def closeEvent(self, event):
        super().closeEvent(event)
        self.close_camera()

    def background(self):
        # 按钮
        self.pushButton.clicked.connect(self.open_camera)  # 打开摄像头
        self.pushButton_2.clicked.connect(self.close_camera)  # 关闭摄像头
        self.pushButton_4.clicked.connect(self.face_recognition.scan_face)

        self.pushButton.setEnabled(True)
        # 初始状态不能关闭摄像头
        self.pushButton_2.setEnabled(False)

    def open_camera(self):
        # 获取选择的设备名称
        index = self.comboBox.currentIndex()
        self.CAM_NUM = index
        # 检测该设备是否能打开
        flag = self.cap.open(self.CAM_NUM)
        if flag is False:
            QMessageBox.information(self, "警告", "该设备未正常连接", QMessageBox.Ok)
        else:
            self.stopEvent.clear()
            th = threading.Thread(target=self.Display)
            th.start()
            # 打开摄像头按钮不能点击
            self.pushButton.setEnabled(False)
            # 关闭摄像头按钮可以点击
            self.pushButton_2.setEnabled(True)

    def Display(self):
        while self.cap.isOpened() and not self.stopEvent.is_set():
            ret, frame = self.cap.read()
            if ret:
                self.lock.acquire()
                self.src_image = copy.deepcopy(frame)
                self.lock.release()

                cur_frame = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2RGB)
                # 视频流的长和宽
                height, width = cur_frame.shape[:2]
                pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(pixmap)
                # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                ratio = max(width / self.label.width(), height / self.label.height())
                pixmap.setDevicePixelRatio(ratio)
                # 视频流置于label中间部分播放
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setPixmap(pixmap)

                cv2.waitKey(1)

        self.stopEvent.clear()

    def close_camera(self):
        self.stopEvent.set()
        self.cap.release()
        self.label.clear()
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = multithread_UI()
    w.show()
    sys.exit(app.exec_())
