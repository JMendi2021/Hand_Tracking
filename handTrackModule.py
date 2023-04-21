import cv2 as cv
import mediapipe as mp


class HandTracker:
    def __init__(self, mode=False, maxHands=2, detection=0.5, model=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection = detection
        self.model = model  # Model Complexity
        self.track = trackCon
        self.results = None
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model, self.detection, self.track)
        self.mpDraw = mp.solutions.drawing_utils

    def finder(self, image, draw=True):  # Draw tells if the landmarks should be drawn or not
        # openCV doesn't work with BGR, so it needs to be converted to RGB
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        # This occurs if it can find
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positions(self, image, handNo=0):
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for landmarkID, lm in enumerate(hand.landmark):  # Loops through all landmarks (0 to 20)
                h, w, c = image.shape  # Returns the height, width and channel of the input image
                landmarks.append([landmarkID, int(lm.x * w), int(lm.y * h)])  # Adds the landmark and its position on the image to the landmark array
        return landmarks



def main():
    cap = cv.VideoCapture(0)  # This establishes a connection with a camera
    tracker = HandTracker()

    while True:
        # Attempts to check if it can read from the camera (Success should be returned if successful
        success, image = cap.read()
        image = tracker.finder(image)

        # Gets the position of each hand landmarks and prints
        landmarks = tracker.positions(image)
        if len(landmarks) != 0:
            print(landmarks)

        cv.imshow("Video", image)
        cv.waitKey(1)  # This allows the video to playback without the needs of a key pressed


if __name__ == "__main__":
    main()
