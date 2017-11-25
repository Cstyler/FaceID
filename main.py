import argparse

import cv2
import dlib

import face_recognition


def show_rects_and_names(detector, recognizer, img, img_gray, haar):
    if haar:
        face_rects = detector.detectMultiScale(img_gray, 1.2, 1, minSize=(10, 10))
        for (x1, y1, w, h) in face_rects:
            x2, y2 = x1 + w, y1 + h
            crop = img[y1:y2, x1:x2]
            w, h, _ = crop.shape
            if w <= 0 or h <= 0:
                return
            person_name = recognizer.predict()
            print(person_name)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    else:
        face_rects = detector(img_gray, 1)
        for (x1, y1, x2, y2) in ((d.left(), d.top(), d.right(), d.bottom()) for d in face_rects):
            crop = img[y1:y2, x1:x2]
            w, h, _ = crop.shape
            if w <= 0 or h <= 0:
                return
            person_name = recognizer.predict(crop)
            print(person_name)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)


def show_image(detector, haar, img_path, recognizer):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_rects_and_names(detector, recognizer, img, img_gray, haar)
    while (True):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return


def show_webcam(detector, haar, video, recognizer):
    cap = cv2.VideoCapture(video if video else 0)

    while (cap.isOpened()):
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        show_rects_and_names(detector, recognizer, img, img_gray, haar)
        cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
        w, h, _ = img.shape
        img = cv2.resize(img, (int(1.5*h), int(1.5*w)))
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_video', type=str, help='input video path')
    parser.add_argument('--in_image', type=str, help='input image path')

    args = parser.parse_args()

    haar = False
    if haar:
        cascade_path = 'haar-face.xml'
        detector = cv2.CascadeClassifier(cascade_path)
    else:
        detector = dlib.get_frontal_face_detector()

    recognizer = face_recognition.Recognizer()
    if args.in_image:
        show_image(detector, haar, args.in_image, recognizer)
    else:
        show_webcam(detector, haar, args.in_video, recognizer)


if __name__ == "__main__":
    main()
