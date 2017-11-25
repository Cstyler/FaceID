
import cv2
import dlib
import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KERAS_BACKEND'] = "theano"
os.environ['THEANO_FLAGS'] = "device=cpu, openmp=true"
os.environ['OMP_NUM_THREADS'] = "2"
import keras


from bottleneck import Bottleneck
from utils import transpose_matrix, Scaler

def init_embeddings(faces_folder: str):
    detector = dlib.get_frontal_face_detector()
    resize_shape = (100, 100)
    cur_label = 0
    labels_dict = dict()
    x, y = [], []
    faces_folder = 'data/MyFaces/'
    for person_folder in os.listdir(faces_folder):
        person_folder = os.path.join(faces_folder, person_folder)
        add_person(x, y, person_folder, cur_label, detector, resize_shape, False)
        labels_dict[cur_label] = person_folder
        cur_label += 1
    x = embed_imgs(x)
    return x, y, labels_dict


def embed_imgs(x):
    npload = np.load('data/mean_std1.npz')
    mean, std = npload['mean'], npload['std']
    scaler = Scaler(mean=mean, std=std)

    model_path = 'data/cnn_model/epoch_232_val_loss1.351451.hdf5'
    model_emb_path = 'data/emb_model/model_1_epoch_0_test_eer0.114874.hdf5'

    # model_path = 'data/cnn_model/epoch_57_val_loss1.699622.hdf5'
    # model_emb_path = 'data/emb_model/model_2_epoch_25_test_eer0.106689.hdf5'


    # model_emb_path = '../data/Modeltpe2/epoch_0_test_eer0.139840.hdf5'
    # model_emb_path = '../data/Modeltpe3/epoch_12_test_eer0.107399.hdf5'
    # model_emb_path = 'data/emb_model/model_4_epoch_1_test_eer0.108006.hdf5'

    model = keras.models.load_model(model_path)
    model_emb = keras.models.load_model(model_emb_path)
    bottleneck = Bottleneck(model)
    x = scaler.transform(x)
    x = bottleneck.predict(transpose_matrix(x))
    x = model_emb.predict(x)
    return x

def get_face_crop(detector, img_gray, img):
    faces = detector(img_gray, 1)
    d = faces[0]
    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
    img = img[y1:y2, x1:x2]
    return img

def add_person(x, y, folder, label, detector, resize_shape, debug=False):
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
            img = get_face_crop(detector, img_gray, img)
        except Exception:
            print('face detector error:', img_path)
            continue
        img = cv2.resize(img, resize_shape)
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