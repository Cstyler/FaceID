import matplotlib.pyplot as plt
import numpy as np
import dlib
import os
import cv2
np.random.seed(42)

import os
os.environ['THEANO_FLAGS'] = "device=cpu, openmp=true"
os.environ['OMP_NUM_THREADS'] = "1"
from keras import backend as K
import keras
from sklearn.neighbors import KNeighborsClassifier


class Scaler(object):
    def __init__(self, x=None, mean=None, std=None):
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(x, axis=0)
        if std is not None:
            self.std = std
        else:
            self.std = np.std(x, axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std


class Bottleneck:
    def __init__(self, model):
        self.fn = K.function([model.layers[0].input, K.learning_phase()], [
                             model.get_layer('bottleneck').output])

    def predict(self, x, batch_size=32, learning_phase=0):
        n_data = len(x)
        n_batches = n_data // batch_size + \
            (0 if n_data % batch_size == 0 else 1)
        output = None
        for i in range(n_batches):
            batch_x = x[i * batch_size:(i + 1) * batch_size]
            batch_y = self.fn([batch_x, learning_phase])[0]

            if output is None:
                output = batch_y
            else:
                output = np.vstack([output, batch_y])
        return output


def transpose_matrix(x):
    return np.transpose(x, (0, 3, 1, 2))


def metric(x, y):
    return -(x @ y.T)

class Recognizer(object):
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.resize_shape = (100, 100)
        npload = np.load('data/mean_std1.npz')
        mean, std = npload['mean'], npload['std']
        self.scaler = Scaler(mean=mean, std=std)

        # model_path = '../data/Model3/epoch_232_val_loss1.351451.hdf5'
        # model_emb_path = '../data/Modeltpe1/epoch_0_test_eer0.114874.hdf5'

        model_path = 'data/cnn_model/epoch_57_val_loss1.699622.hdf5'
        # model_emb_path = '../data/Modeltpe2/epoch_0_test_eer0.139840.hdf5'
        # model_emb_path = '../data/Modeltpe2/epoch_25_test_eer0.106689.hdf5'
        # model_emb_path = '../data/Modeltpe3/epoch_12_test_eer0.107399.hdf5'
        model_emb_path = 'data/emb_model/model_4_epoch_1_test_eer0.108006.hdf5'

        model = keras.models.load_model(model_path)
        self.model_emb = keras.models.load_model(model_emb_path)
        self.bottleneck = Bottleneck(model)
        faces_folder = '/home/agazade/FaceID/datasets/MyFaces/'
        self.x, self.y = [], []
        self.init_embeddings(faces_folder)

        self.knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
        self.knn.fit(self.x, self.y)

    def init_embeddings(self, faces_folder):
        cur_label = 0
        self.labels_dict = dict()
        for person_folder in os.listdir(faces_folder):
            self.add_person(os.path.join(faces_folder, person_folder), cur_label, 0)
            self.labels_dict[cur_label] = person_folder
            cur_label += 1
        self.x, self.y = np.array(self.x), np.array(self.y)

    def add_person(self, folder, label, debug=False):
        batch_x = []
        formats = {'.jpg', '.jpeg', '.png'}
        for img_path in os.listdir(folder):
            root, ext = os.path.splitext(img_path)
            if ext not in formats:
                continue
            img_path = os.path.join(folder, img_path)
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is None:
                print('error img:', img_path)
                continue
            try:
                img = self.crop_img(img_gray, img)
            except Exception as e:
                print('face detector error:', img_path)
                continue
            img = cv2.resize(img, self.resize_shape)
            if debug:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            batch_x.append(img)
            self.y.append(label)
        batch_x = self.scaler.transform(batch_x)
        batch_x = self.bottleneck.predict(transpose_matrix(batch_x))
        batch_x = self.model_emb.predict(batch_x)
        self.x.extend(batch_x)

    def crop_img(self, img_gray, img):
        dets = self.detector(img_gray, 1)
        d = dets[0]
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        img = img[y1:y2, x1:x2]
        return img

    def predict(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            img = self.crop_img(img_gray, img)
        except Exception as e:
            print('face detector error')
            return
        img = cv2.resize(img, self.resize_shape)
        batch_x = [img]
        batch_x = self.scaler.transform(batch_x)
        batch_x = self.bottleneck.predict(transpose_matrix(batch_x))
        batch_x = self.model_emb.predict(batch_x)

        pred_labels = self.knn.predict(batch_x)
        label = pred_labels[0]
        return self.labels_dict[label]


