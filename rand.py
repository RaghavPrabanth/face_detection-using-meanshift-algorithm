import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(r"C:\Users\Raghav Prabanth\OneDrive\Desktop\python learner files\haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if ret == True:
        # Detect faces in every frame
        face_rects = face_cascade.detectMultiScale(frame)

        if len(face_rects) > 0:
            # Get the first face rectangle
            (face_x, face_y, w, h) = tuple(face_rects[0])
            track_window = (face_x, face_y, w, h)

            # Convert the face region to HSV
            roi = frame[face_y:face_y+h, face_x:face_x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Calculate the histogram of the HSV image
            roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Define the termination criteria
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            # Convert the frame to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Calculate the backprojection of the histogram
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # Apply the meanShift algorithm
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw the face rectangle
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

            cv2.imshow('img', img2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()