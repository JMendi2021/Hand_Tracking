import cv2 as cv
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
cap = cv.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read() # Checks if it can read from the camera

    # Mediapipe does not work with BGR images so needs to be converted to RGB
    imageRDB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(imageRDB)  # This processes the frame and is used for detecting the hand

    if results.multi_hand_landmarks: #This occurs if there is a hand or two present
        for hand in results.multi_hand_landmarks:  # Works for each hand present on the screen
            mpDraw.draw_landmarks(image, hand,
                                  mpHands.HAND_CONNECTIONS)  # Displays a connected landmark in the detected hand

    # Displays the image
    cv.imshow("Camera", image)
    cv.waitKey(1)
