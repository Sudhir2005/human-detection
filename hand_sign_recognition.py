import cv2

def capture_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
    
########################################################################################################################################
import cv2
import mediapipe as mp

def capture_video():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()

################################################################################################################################################
def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    count = 0
    if landmarks[4].y < landmarks[3].y:  # Thumb
        count += 1
    if landmarks[8].y < landmarks[6].y:  # Index
        count += 1
    if landmarks[12].y < landmarks[10].y:  # Middle
        count += 1
    if landmarks[16].y < landmarks[14].y:  # Ring
        count += 1
    if landmarks[20].y < landmarks[18].y:  # Pinky
        count += 1
    return count

################################################################################################################################################

import cv2
import mediapipe as mp

def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    count = 0
    if landmarks[4].y < landmarks[3].y:  # Thumb
        count += 1
    if landmarks[8].y < landmarks[6].y:  # Index
        count += 1
    if landmarks[12].y < landmarks[10].y:  # Middle
        count += 1
    if landmarks[16].y < landmarks[14].y:  # Ring
        count += 1
    if landmarks[20].y < landmarks[18].y:  # Pinky
        count += 1
    return count

def capture_video():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(hand_landmarks)
                cv2.putText(frame, str(fingers), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()

