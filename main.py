import sys
import typing
from PyQt5 import QtCore, QtGui
import copy
import cv2
import numpy as np
import os
import shutil

from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit, QWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from resource.UI.Ui_face_recognition_gui import Ui_MainWindow

from knn_webcam import train, predict
import face_recognition


class face_recognition_UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.camera = cv2.VideoCapture()

        self.system_state_lock = (
            0  # 标志系统状态的量 0表示无子线程在运行 1表示调用摄像头 2表示正在人脸识别 3表示正在录入新面孔 4表示正在训练新面孔。
        )

        self.background()

    def closeEvent(self, event):
        super().closeEvent(event)
        self.close_camera()

    def background(self):
        # 按钮
        self.pushButton.clicked.connect(self.open_camera)  # 打开摄像头
        self.pushButton_2.clicked.connect(self.close_camera)  # 关闭摄像头
        self.pushButton_3.clicked.connect(self.f_get_new_face)  # 录入新人脸
        self.pushButton_4.clicked.connect(self.f_scan_face)  # 人脸识别
        self.pushButton_5.clicked.connect(self.stop_scan_face)  # 退出识别

        self.init_UI_state()

    def init_UI_state(self):
        """UI初始化"""
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.label.clear()

        # TODO:可以在界面最顶部加个显示当前状态，while True

    def open_camera(self):
        # 获取选择的设备名称
        if self.system_state_lock == 0:
            index = self.comboBox.currentIndex()
            self.CAM_NUM = index
            # 检测该设备是否能打开
            flag = self.camera.open(self.CAM_NUM)
            if flag is False:
                QMessageBox.warning(self, "警告", "该设备未正常连接", QMessageBox.Ok)
            else:
                self.pushButton.setEnabled(False)
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(True)
                self.system_state_lock = 1
                self.display_camera()

    def display_camera(self):
        while self.camera.isOpened() and self.system_state_lock == 1:
            ret, frame = self.camera.read()
            if ret:
                cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    def f_scan_face(self):
        def scan_face(
            self,
            model_path="knn_examples/trained_knn_model.clf",
            distance_threshold=0.38,
        ):
            """不断调用摄像头扫描并做人脸识别"""
            if not os.path.isfile(model_path):
                QMessageBox.warning(self, "警告", "系统未收集人脸信息", QMessageBox.Ok)
                self.system_state_lock = 1
            else:
                self.pushButton_4.setEnabled(False)
                self.pushButton_5.setEnabled(True)

            while self.camera.isOpened() and self.system_state_lock == 2:
                # Grab a single frame of video
                ret, frame = self.camera.read()

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face enqcodings in the frame of video
                predictions = predict(
                    cur_frame,
                    model_path=model_path,
                    distance_threshold=distance_threshold,
                )

                # Loop through each face in this frame of video
                for name, (top, right, bottom, left) in predictions:
                    # Draw a box around the face
                    cv2.rectangle(
                        cur_frame, (left, top), (right, bottom), (0, 0, 255), 2
                    )

                    # Draw a label with a name below the face
                    cv2.rectangle(
                        cur_frame,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(
                        cur_frame,
                        name,
                        (left + 6, bottom - 6),
                        font,
                        1.0,
                        (255, 255, 255),
                        1,
                    )

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

        self.system_state_lock = 2
        scan_face(self)

    def stop_scan_face(self):
        self.system_state_lock = 1
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(False)
        self.display_camera()

    def f_get_new_face(self):
        def get_new_face(self, filepath="knn_examples/train", total_num=10, name=None):
            """录入人脸照片

            Args:
                filepath (str, optional): 存放照片的路径. Defaults to 'knn_examples/train'.
                total_num (int, optional): 拍摄总数. Defaults to 100.
                name (str, optional): 录入人脸姓名. Defaults to None.
            """

            print("正在从摄像头录入新人脸信息 \n")

            # 存在目录就清空，不存在就创建，确保最后存在空的目录
            filepath = os.path.join(filepath, name)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            else:
                shutil.rmtree(filepath)
                os.makedirs(filepath)

            sample_num = 0  # 已经获得的样本数

            while self.camera.isOpened():
                success, frame = self.camera.read()
                # 保存的是没加框的图片
                cv2.imwrite(os.path.join(filepath, str(sample_num) + ".jpg"), frame)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_bounding_boxes = face_recognition.face_locations(frame)
                # 框选人脸，for循环保证一个能检测的实时动态视频流
                for face_bounding_box in face_bounding_boxes:
                    top, right, bottom, left = face_bounding_box
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    sample_num += 1

                height, width = frame.shape[:2]
                pixmap = QImage(frame, width, height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(pixmap)
                # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                ratio = max(width / self.label.width(), height / self.label.height())
                pixmap.setDevicePixelRatio(ratio)
                # 视频流置于label中间部分播放
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setPixmap(pixmap)

                cv2.waitKey(1)

                if sample_num >= total_num:
                    print("录入结束")
                    break

        def train_new_face(
            self,
            train_dir="knn_examples/train",
            model_save_path="knn_examples/trained_knn_model.clf",
            n_neighbors=2,
        ):
            """对新录入的人脸训练一个KNN分类器

            Args:
                train_dir (str, optional): 新录入的人脸存放位置. Defaults to "knn_examples/train".
                model_save_path (str, optional): 保存模型的位置. Defaults to "knn_examples/trained_knn_model.clf".
                n_neighbors (int, optional): KNN参数. Defaults to 2.

            Returns:
                _type_: KNN分类器
            """

            print("正在训练KNN分类器：")
            classifier = train(
                train_dir=train_dir,
                model_save_path=model_save_path,
                n_neighbors=n_neighbors,
            )
            print("KNN分类器训练完成！")
            return classifier

        name, have_user_name = QInputDialog.getText(
            self, "录入人脸", "请输入姓名:", QLineEdit.Normal, ""
        )
        if not have_user_name:
            return

        self.system_state_lock = 3
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        get_new_face(self, name=name)

        self.system_state_lock = 4
        train_new_face(self)

        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(False)

        QMessageBox.information(self, "人脸采集", "新人脸录入完成", QMessageBox.Ok)

        self.system_state_lock = 1
        self.display_camera()

    def close_camera(self):
        self.system_state_lock = 0
        self.camera.release()
        self.init_UI_state()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = face_recognition_UI()
    w.show()
    sys.exit(app.exec_())
