import os
import pickle

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['KERAS_BACKEND'] = "theano"
os.environ['THEANO_FLAGS'] = "device=cpu, openmp=true"
os.environ['OMP_NUM_THREADS'] = "2"
import keras

from bottleneck import Bottleneck
from utils import transpose_matrix, \
    Scaler, RecognitionError, \
    align_and_crop_img, get_template_landmarks


def init_embeddings(faces_folder: str):
    resize_shape = (100, 100)
    detector = dlib.get_frontal_face_detector()
    shape_predictor_path = 'data/shape_predictor_68_face_landmarks.dat'
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    eye_and_mouth_indices = [39, 42, 57]
    template_landmarks = get_template_landmarks(
        eye_and_mouth_indices, resize_shape[0])
    cur_label = 0
    labels_dict = dict()
    x, y = [], []
    faces_folder = 'data/MyFaces/'
    debug = False
    for person_folder in os.listdir(faces_folder):
        labels_dict[cur_label] = person_folder
        person_folder = os.path.join(faces_folder, person_folder)
        add_person(x, y, person_folder, cur_label,
                   detector, shape_predictor,
                   template_landmarks, eye_and_mouth_indices, resize_shape, debug)
        cur_label += 1
    x = embed_imgs(x)
    return x, y, labels_dict


def embed_imgs(x):
    npload = np.load('data/mean_std2.npz')
    mean, std = npload['mean'], npload['std']
    scaler = Scaler(mean=mean, std=std)

    # model_path = 'data/cnn_model/epoch_232_val_loss1.351451.hdf5'
    # model_emb_path = 'data/emb_model/model_1_epoch_0_test_eer0.114874.hdf5'

    # model_path = 'data/cnn_model/epoch_57_val_loss1.699622.hdf5'
    # model_emb_path = 'data/emb_model/model_2_epoch_25_test_eer0.106689.hdf5'

    # model_path = 'data/cnn_model/epoch_29_val_loss1.441430.hdf5'
    # model_emb_path = 'data/emb_model/model_5_epoch_2_test_eer0.143211.hdf5'
    # model_emb_path = 'data/emb_model/model_6_epoch_6_test_eer_0.135497_test2_err0.254601.hdf5'

    # model_emb_path = '../data/Modeltpe2/epoch_0_test_eer0.139840.hdf5'
    # model_emb_path = '../data/Modeltpe3/epoch_12_test_eer0.107399.hdf5'
    # model_emb_path = 'data/emb_model/model_4_epoch_1_test_eer0.108006.hdf5'

    # model_path = 'data/cnn_model/epoch_16_val_loss1.231896.hdf5'
    # model_emb_path = 'data/emb_model/model_8_epoch_15_test_eer0.127431_test2_err0.218662.hdf5'
    # model_emb_path = 'data/emb_model/model_8_epoch_1_test_eer0.133520_test2_err0.216839.hdf5'
    # model_emb_path = 'data/emb_model/model_9_epoch_5_test_eer0.127574_test2_err0.229637.hdf5'

    model_path = 'data/cnn_model/epoch_66_val_loss1.206078.hdf5'
    model_emb_path = 'data/emb_model/model_10_epoch_10_test_eer0.169731_test2_err0.204908.hdf5'

    model = keras.models.load_model(model_path)
    model_emb = keras.models.load_model(model_emb_path)
    bottleneck = Bottleneck(model)
    x = scaler.transform(x)
    x = bottleneck.predict(transpose_matrix(x))
    x = model_emb.predict(x)
    return x


def add_person(x, y, folder, label,
               detector, shape_predictor,
               template_landmarks, landmark_indices, resize_shape, debug):
    formats = {'.jpg', '.jpeg', '.png'}
    for img_path in os.listdir(folder):
        root, ext = os.path.splitext(img_path)
        if ext not in formats:
            continue
        img_path = os.path.join(folder, img_path)
        img = cv2.imread(img_path)
        if img is None:
            print('error img:', img_path)
            continue
        try:
            img = align_and_crop_img(img, resize_shape, detector, shape_predictor, template_landmarks,
                                     landmark_indices)
        except RecognitionError:
            print('face detector error:', img_path)
            continue
        if debug:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        x.append(img)
        y.append(label)


def main():
    main_folder = 'data'
    faces_folder = os.path.join(main_folder, 'MyFaces')
    x, y, labels_dict = init_embeddings(faces_folder)
    np.savez(os.path.join(main_folder, 'face_base'), x=x, y=y)
    with open(os.path.join(main_folder, 'labels_dict.pkl'), 'wb') as file:
        pickle.dump(labels_dict, file)


if __name__ == '__main__':
    main()
