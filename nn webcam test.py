#Real time webcam NN model test
import keras
from keras import layers
from keras import models
import numpy as np
from time import sleep
import cv2
#Use the same from either converting function to create matrix from hand_landmark mappings:
mediapipe2posenet = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
def convertPoseMapToArray(pose_landmarks):

        #Input: pose landmark mapping made of 17 different points on body
       # Output: 51 dimension x-vector, x,y coordinates and visibily of each the 17 points in an array
    
    global mediapipe2posenet
    x = np.zeros(51) #output array
    for i in range(17): #for the 17 land marks put the x,y,z coordinates and visibility in the output array
        x[i*3+0] = pose_landmarks.landmark[mediapipe2posenet[i]].x
        x[i*3+1] = pose_landmarks.landmark[mediapipe2posenet[i]].y
        x[i*3+2] = pose_landmarks.landmark[mediapipe2posenet[i]].visibility
    return x

def convertPoseMapToArrayRelative(pose_landmarks):
    
       # Input: pose landmark mapping made of 17 different points on body
       # Output: 51 dimension x-vector, x,y coordinates minus nose coordinate, and visibily of each the 17 points in an array
    
    global mediapipe2posenet
    x_nose = pose_landmarks.landmark[0].x
    y_nose = pose_landmarks.landmark[0].y
    x = np.zeros(51) #output array
    for i in range(17): #for the 17 land marks put the x,y,z coordinates and visibility in the output array
        x[i*3+0] = pose_landmarks.landmark[mediapipe2posenet[i]].x - x_nose
        x[i*3+1] = pose_landmarks.landmark[mediapipe2posenet[i]].y - y_nose
        x[i*3+2] = pose_landmarks.landmark[mediapipe2posenet[i]].visibility

    return x
#Load model from file:
reconstructed_model = keras.models.load_model('push_up_catchallModel')
t#his is a simple web cam example, with no hand mapping
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

#     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#     IPython.display.Image(frame)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
#This is the webcam with the NN processing
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale = 1
fontColor = (255,255,255)
lineType = 2
output = 0

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            x = convertPoseMapToArrayRelative(results.pose_landmarks)
            if x.all() != None:
                output = np.argmax(reconstructed_model.predict(x.reshape((1, 51))))
#         mp_drawing.draw_landmarks(
#             image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#         mp_drawing.draw_landmarks(
#             image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#         mp_drawing.draw_landmarks(
#             image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
         ###############################################
        # This is where the text on screen is displayed:
        ###############################################
        cv2.putText(image,str(output), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        ###############################################
        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()