import numpy as np
import cv2
import skimage.transform as tr


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


def transpose_matrix(x):
    return np.transpose(x, (0, 3, 1, 2))


def points_to_np(shape):
    return np.array([[p.x, p.y] for p in shape.parts()])


class RecognitionError(BaseException):
    pass


def align_and_crop_img(img, output_shape, detector, shape_predictor, template_landmarks, landmarks_indices):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    if len(dets) != 1:
        raise RecognitionError()
    d = dets[0]
    aligned_img = align_img(img, img_gray, d, shape_predictor, template_landmarks, landmarks_indices, output_shape)
    return aligned_img


def align_img(img, img_gray, rect, shape_predictor, template_landmarks, landmark_indices, output_shape):
    pred_landmarks = points_to_np(shape_predictor(img_gray, rect))
    pred_landmarks = pred_landmarks[landmark_indices]

    ones = np.ones((3, 1))
    A = np.hstack((pred_landmarks, ones))
    B = np.hstack((template_landmarks, ones))
    T = np.linalg.solve(A, B).T  # (T@A.T).T = B

    wrapped = tr.warp(img, tr.AffineTransform(T).inverse,
                      output_shape=output_shape, preserve_range=False)
    wrapped = np.uint8(wrapped * 255)
    #     plt.imshow(cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB))
    #     plt.show()
    return wrapped


def get_template_landmarks(indices, resize_factor):
    face_template_path = 'data/face_template.npy'
    face_template = np.load(face_template_path)
    template_key_points = resize_factor * \
                          face_template[indices]
    return template_key_points
