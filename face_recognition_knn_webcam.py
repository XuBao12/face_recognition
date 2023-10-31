"""
完全体代码。
构造了用于识别的face_recognition_gui类，在类内实现了录入照片、KNN分类器的训练、识别人脸三个功能。
初始化该类时传入cv2调用摄像头得到的camera，拍摄的每帧图片通过成员变量self.frame进行全局传递和感知。
"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
import shutil
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import numpy as np

from knn_webcam import train, predict

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


class face_recognition_gui(object):
    def __init__(self, camera: cv2.VideoCapture) -> None:
        self.camera = camera
        self.success, self.frame = camera.read()

    def get_new_face(self, filepath="data", total_num=100, name=None):
        """录入人脸照片

        Args:
            filepath (str, optional): 存放照片的路径. Defaults to 'data'.
            total_num (int, optional): 拍摄总数. Defaults to 100.
            name (str, optional): 录入人脸姓名. Defaults to None.
        """

        print("正在从摄像头录入新人脸信息 \n")

        # 存在目录就清空，不存在就创建，确保最后存在空的目录
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            shutil.rmtree(filepath)
            os.mkdir(filepath)

        sample_num = 0  # 已经获得的样本数

        while True:
            frame = self.frame.copy()
            face_bounding_boxes = face_recognition.face_locations(frame)

            # 框选人脸，for循环保证一个能检测的实时动态视频流
            for face_bounding_box in face_bounding_boxes:
                top, right, bottom, left = face_bounding_box
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                sample_num += 1

                # 保存的是没加框的图片
                cv2.imwrite(
                    filepath + "/" + name + "." + str(sample_num) + ".jpg", self.frame
                )

            cv2.imshow("Video", frame)

            cv2.waitKey(1)
            if sample_num > total_num:
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

    def scan_face(self, model_path="knn_examples/trained_knn_model.clf"):
        """不断调用摄像头扫描并做人脸识别，按q退出

        Args:
            model_path (str, optional): 存放KNN分类器的位置. Defaults to "knn_examples/trained_knn_model.clf".
        """

        print("正在扫描人脸：")

        while True:
            # Grab a single frame of video
            frame = self.frame.copy()

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face enqcodings in the frame of video
            predictions = predict(rgb_frame, model_path=model_path)

            # Loop through each face in this frame of video
            for name, (top, right, bottom, left) in predictions:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
                )

            # Display the resulting image
            cv2.imshow("Video", frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def destroy(self):
        """释放摄像头"""
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)

    gui = face_recognition_gui(video_capture)
    gui.scan_face()
    gui.destroy()
