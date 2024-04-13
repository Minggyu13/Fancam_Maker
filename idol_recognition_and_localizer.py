import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

#얼굴 추출과 face_recognition을 위한 코드 입니다.

faces_embeddings = np.load("face_embeddings_done_6classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open("fancam_svm_model.pkl", 'rb'))


class idol_classifier:

    def __init__(self, file_path):
        self.detector = MTCNN()
        self.file_path = file_path
        self.embedder = FaceNet()
        self.target_size = (160, 160)


    def get_embedding(self, face_img):
        fece_img = face_img.astype('float32')  # 3D (160 x 160 x 3)
        face_img = np.expand_dims(face_img, axis=0)  # 4D (none x 160 x 160 x 3)
        yhat = self.embedder.embeddings(face_img)

        return yhat[0]  # 512D (1 x 1 x 512)

    def extract_face(self):
        img = cv.imread(self.file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        face_detection = self.detector.detect_faces(img)  # x, y, w, h
        face_box_arr = list()
        resized_face_arr = list()
        for i in range(len(face_detection)):
            face_box = face_detection[i]['box']
            x, y = abs(face_box[0]), abs(face_box[1])
            face = img[y: y + face_box[3], x: x + face_box[2]]
            resized_face = cv.resize(face, self.target_size)
            face_box_arr.append(face_box)
            resized_face_arr.append(resized_face)
        return resized_face_arr, face_box_arr

    def svm_infer(self, extracted_face):
        # extracted_face = self.extract_face()
        embedded_face = self.get_embedding(extracted_face)

        face_name = model.predict(np.array([embedded_face]))
        idol_name = encoder.inverse_transform(face_name)[0]



        return idol_name
#
#
# idol = idol_classifier('oneyoung_test1.jpg')
#
# face = idol.extract_face()
#
# reszi = face[0]
# label = idol.svm_infer(reszi[0])
# print(label)
#


