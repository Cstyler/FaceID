import argparse

import cv2
import dlib
import face_recognition


def show_rects(detector, img, haar):
    if haar:
        dets = detector.detectMultiScale(img, 1.2, 1, minSize=(10, 10))
        for (x1, y1, w, h) in dets:
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
    else:
        dets = detector(img, 1)
        for (x1, y1, x2, y2) in ((d.left(), d.top(), d.right(), d.bottom()) for d in dets):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # print(len(dets))


def show_image(detector, haar, image, recognizer):
    img = cv2.imread(image, 0)
    show_rects(detector, img, haar)
    while (True):
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return


def show_webcam(detector, haar, video, recognizer):
    cap = cv2.VideoCapture(video if video else 0)

    while (cap.isOpened()):
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[::]

        show_rects(detector, img, haar)
        print(recognizer.predict(img))
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_video', type=str, help='input video path')
    parser.add_argument('--in_image', type=str, help='input image path')


    args = parser.parse_args()

    cascade_path = 'haar-face.xml'
    haar = False
    if haar:
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
