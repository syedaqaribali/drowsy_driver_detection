import cv2
import dlib

# Load the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
from scipy.spatial import distance as dist
# Get the webcam
cap = cv2.VideoCapture(0)
def aspect_ratio(eye):
    A = dist.euclidean(eye[0:2], eye[2:4])
    B = dist.euclidean(eye[0:2], eye[2:4])
    ear = A / B
    return ear


# define the threshold for closed eyes


# define the threshold for closed eyes
EYE_AR_THRESH_CLOSE = 0.1
EYE_AR_THRESH_OPEN = 0.2
COUNTER = 0
# loop to detect drowsiness
while True:
    # Read the webcam image
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        landmarks = predictor(gray,dlib.rectangle(x,y,x+w,y+h))
        left_eye = (landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(39).y)
        right_eye = (landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        left_eye_aspect_ratio = aspect_ratio(left_eye)
        right_eye_aspect_ratio = aspect_ratio(right_eye)
        eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
        
        cv2.putText(img, "EAR: {:.2f}".format(eye_aspect_ratio), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # detect facial landmarks
            landmarks = predictor(gray,dlib.rectangle(x,y,x+w,y+h))
            left_eye = (landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(39).y)
            right_eye = (landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y)
            left_eye_aspect_ratio = aspect_ratio(left_eye)
            right_eye_aspect_ratio = aspect_ratio(right_eye)
            eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
            #cv2.putText(img, "EAR: {:.2f}".format(eye_aspect_ratio), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if eye_aspect_ratio < EYE_AR_THRESH_CLOSE:
                # Eye is closed
                cv2.putText(img, "Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER += 1
                if COUNTER >= 30:
                    cv2.putText(img, "Drowsy", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif EYE_AR_THRESH_CLOSE <= eye_aspect_ratio < EYE_AR_THRESH_OPEN:
                # Eye is Half open
                cv2.putText(img, "Half Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER += 1
                if COUNTER >= 30:
                    cv2.putText(img, "Drowsy", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Eye is open
                cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER = 0

    # Show the image
    cv2.imshow('img', img)

    # Exit the webcam if 'q' is pressed
    if cv2.waitKey(25) == 27:
        break


# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()



