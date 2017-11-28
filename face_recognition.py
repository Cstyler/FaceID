import dlib
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
from utils import get_template_landmarks, align_img


def metric(x, y):
    return -(x @ y.T)


class Recognizer(object):
    def __init__(self):
        self.resize_shape = (100, 100)
        shape_predictor_path = 'data/shape_predictor_68_face_landmarks.dat'
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.eye_and_mouth_indices = [39, 42, 57]
        self.template_landmarks = get_template_landmarks(
            self.eye_and_mouth_indices, self.resize_shape[0])
        npload = np.load('data/mean_std2.npz')
        mean, std = npload['mean'], npload['std']
        self.scaler = Scaler(mean=mean, std=std)

        model_path = 'data/cnn_model/epoch_66_val_loss1.206078.hdf5'
        model_emb_path = 'data/emb_model/model_10_epoch_10_test_eer0.169731_test2_err0.204908.hdf5'

        # model_path = 'data/cnn_model/epoch_16_val_loss1.231896.hdf5'
        # model_emb_path = 'data/emb_model/model_8_epoch_15_test_eer0.127431_test2_err0.218662.hdf5'
        # model_emb_path = 'data/emb_model/model_8_epoch_1_test_eer0.133520_test2_err0.216839.hdf5'
        # model_emb_path = 'data/emb_model/model_9_epoch_5_test_eer0.127574_test2_err0.229637.hdf5'


        # model_path = 'data/cnn_model/epoch_232_val_loss1.351451.hdf5'
        # model_emb_path = 'data/emb_model/model_1_epoch_0_test_eer0.114874.hdf5'
        #
        # model_path = 'data/cnn_model/epoch_57_val_loss1.699622.hdf5'
        # model_emb_path = 'data/emb_model/model_2_epoch_25_test_eer0.106689.hdf5'

        # model_path = 'data/cnn_model/epoch_29_val_loss1.441430.hdf5'
        # model_emb_path = 'data/emb_model/model_5_epoch_2_test_eer0.143211.hdf5'
        # model_emb_path = 'data/emb_model/model_6_epoch_6_test_eer_0.135497_test2_err0.254601.hdf5'

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

        self.knn = KNeighborsClassifier(n_neighbors=1, metric=metric, n_jobs=1)
        self.knn.fit(self.x, self.y)

    def iterate_similarities(self, emb):
        for i, person_emb in enumerate(self.x):
            sim = person_emb @ emb.T
            yield sim, i

    def predict(self, img, img_gray, rect):
        img = align_img(img, img_gray, rect, self.shape_predictor, self.template_landmarks,
                        self.eye_and_mouth_indices, self.resize_shape)
        batch_x = [img]
        batch_x = self.scaler.transform(batch_x)
        batch_x = self.bottleneck.predict(transpose_matrix(batch_x))
        batch_x = self.model_emb.predict(batch_x)

        pred_labels = self.knn.predict(batch_x)
        neighbors = self.knn.kneighbors(batch_x)
        label_neighbors = [self.labels_dict[self.y[ind]] for ind in neighbors[1][0]]
        # print(label_neighbors, neighbors[0])

        # label_ind = max(self.iterate_similarities(batch_x[0]), key=lambda x: x[0])[1]
        # label = self.y[label_ind]
        label = pred_labels[0]
        return self.labels_dict[label], label_neighbors
