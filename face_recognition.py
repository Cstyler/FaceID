import cv2
import dlib
import matplotlib.pyplot as plt
import pickle
import numpy as np

np.random.seed(42)

import os
os.environ['KERAS_BACKEND'] = "theano"
os.environ['THEANO_FLAGS'] = "device=cpu, openmp=true"
os.environ['OMP_NUM_THREADS'] = "1"
import keras
from sklearn.neighbors import KNeighborsClassifier
from utils import Scaler, transpose_matrix
from bottleneck import Bottleneck


def metric(x, y):
    return -(x @ y.T)


class Recognizer(object):
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.resize_shape = (100, 100)
        npload = np.load('data/mean_std1.npz')
        mean, std = npload['mean'], npload['std']
        self.scaler = Scaler(mean=mean, std=std)

        model_path = 'data/cnn_model/epoch_232_val_loss1.351451.hdf5'
        model_emb_path = 'data/emb_model/model_1_epoch_0_test_eer0.114874.hdf5'

        # model_path = 'data/cnn_model/epoch_57_val_loss1.699622.hdf5'
        # model_emb_path = 'data/emb_model/model_2_epoch_25_test_eer0.106689.hdf5'

        # model_emb_path = '../data/Modeltpe2/epoch_0_test_eer0.139840.hdf5'
        # model_emb_path = '../data/Modeltpe3/epoch_12_test_eer0.107399.hdf5'
        # model_emb_path = 'data/emb_model/model_4_epoch_1_test_eer0.108006.hdf5'

        model = keras.models.load_model(model_path)
        self.model_emb = keras.models.load_model(model_emb_path)
        self.bottleneck = Bottleneck(model)

        npload = np.load('data/face_base.npz')
        self.x, self.y = npload['x'], npload['y']
        print(self.x.shape, self.y.shape)

        with open('data/labels_dict.pkl', 'rb') as file:
            self.labels_dict = pickle.load(file)

        self.knn = KNeighborsClassifier(n_neighbors=3, metric=metric, n_jobs=1)
        self.knn.fit(self.x, self.y)


    def predict(self, img):
        img = cv2.resize(img, self.resize_shape)
        batch_x = [img]
        batch_x = self.scaler.transform(batch_x)
        batch_x = self.bottleneck.predict(transpose_matrix(batch_x))
        batch_x = self.model_emb.predict(batch_x)

        pred_labels = self.knn.predict(batch_x)
        neighbors = self.knn.kneighbors(batch_x)
        label_neighbors = [self.labels_dict[self.y[ind]] for ind in neighbors[1][0]]
        label = pred_labels[0]
        return self.labels_dict[label], label_neighbors
