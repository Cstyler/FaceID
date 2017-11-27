import argparse

import cv2
import dlib

import face_recognition

def draw_text(img, text, x1, y1, thickness):
    font_size = 1
    cv2.putText(img, text, (x1 - thickness, y1 - thickness),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness // 4, cv2.LINE_AA)

def show_rects_and_names(detector, recognizer, img, img_gray, haar):
    if haar:
        face_rects = detector.detectMultiScale(img_gray, 1.2, 1, minSize=(10, 10))
        for (x1, y1, w, h) in face_rects:
            x2, y2 = x1 + w, y1 + h
            crop = img[y1:y2, x1:x2]
            w, h, _ = crop.shape
            if w <= 0 or h <= 0:
                return
            thickness = 2
            person_name = recognizer.predict(crop)
            draw_text(img, person_name[0], x1, y1, thickness)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)
    else:
        face_rects = detector(img_gray, 1)
        for (x1, y1, x2, y2) in ((d.left(), d.top(), d.right(), d.bottom()) for d in face_rects):
            crop = img[y1:y2, x1:x2]
            w, h, _ = crop.shape
            if w <= 0 or h <= 0:
                return
            person_name = recognizer.predict(crop)
            thickness = 2
            draw_text(img, person_name[0], x1, y1, thickness)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)


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


def show_webcam(detector, haar, video, recognizer, factor):
    cap = cv2.VideoCapture(video if video else 0)
    if video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('videos/output.avi', fourcc, 20.0, (640, 480), True)
    while (cap.isOpened()):
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        show_rects_and_names(detector, recognizer, img, img_gray, haar)
        w, h, _ = img.shape

        img = cv2.resize(img, (int(factor*h), int(factor*w)))
        if video:
            img = cv2.resize(img, (640, 480))
            out.write(img)
            pass
        else:
            cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video:
        out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_video', type=str, help='input video path')
    parser.add_argument('--in_image', type=str, help='input image path')
    parser.add_argument('--big', help='big_window', action='store_true')

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
        if args.big:
            factor = 1.5
        else:
            factor = 1.0
        show_webcam(detector, haar, args.in_video, recognizer, factor)


if __name__ == "__main__":
    main()
