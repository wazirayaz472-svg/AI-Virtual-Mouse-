import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# ------------------- Settings & Calibration -------------------
w_cam, h_cam = 640, 480
screen_width, screen_height = pyautogui.size()
frame_reduction = 100  # Margin for easier screen edge reaching
smoothening = 7        # Higher = smoother but more lag
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Disable PyAutoGUI fail-safe to prevent crashes if cursor hits corners
pyautogui.FAILSAFE = False

# ------------------- Mediapipe Setup -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ------------------- Webcam Setup -------------------
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw the "Active Area" boundary
    cv2.rectangle(frame, (frame_reduction, frame_reduction), 
                  (w_cam - frame_reduction, h_cam - frame_reduction), (255, 0, 255), 2)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract Landmark positions
        lm_list = []
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = frame.shape
            lm_list.append((int(lm.x * w), int(lm.y * h)))

        if len(lm_list) != 0:
            # Tip of Index (8) and Middle (12) and Thumb (4)
            x1, y1 = lm_list[8]
            x2, y2 = lm_list[12]
            xt, yt = lm_list[4]

            # Check which fingers are up
            # (MediaPipe Landmarks: 8 is index tip, 6 is index pip)
            index_up = y1 < lm_list[6][1]
            middle_up = y2 < lm_list[10][1]

            # 1. MOVEMENT MODE: Only Index Finger is Up
            if index_up and not middle_up:
                # Convert Coordinates using Numpy Interp
                # This maps the inner rectangle to the full screen resolution
                x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, screen_width))
                y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, screen_height))

                # Smoothening logic
                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y

            # 2. LEFT CLICK: Index and Middle are both up and close
            if index_up and middle_up:
                dist = get_distance((x1, y1), (x2, y2))
                if dist < 40:
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.1) # Small delay to prevent click spam

            # 3. RIGHT CLICK: Thumb and Index Finger pinch
            dist_right = get_distance((x1, y1), (xt, yt))
            if dist_right < 40:
                cv2.circle(frame, (xt, yt), 15, (0, 0, 255), cv2.FILLED)
                pyautogui.rightClick()
                time.sleep(0.2)

    # Display FPS (Optional)
    cv2.imshow("AI Virtual Mouse", frame)

    # Exit on 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# ------------------- Settings & Calibration -------------------
w_cam, h_cam = 640, 480
screen_width, screen_height = pyautogui.size()
frame_reduction = 100  # Margin for easier screen edge reaching
smoothening = 7        # Higher = smoother but more lag
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Disable PyAutoGUI fail-safe to prevent crashes if cursor hits corners
pyautogui.FAILSAFE = False

# ------------------- Mediapipe Setup -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ------------------- Webcam Setup -------------------
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw the "Active Area" boundary
    cv2.rectangle(frame, (frame_reduction, frame_reduction), 
                  (w_cam - frame_reduction, h_cam - frame_reduction), (255, 0, 255), 2)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract Landmark positions
        lm_list = []
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = frame.shape
            lm_list.append((int(lm.x * w), int(lm.y * h)))

        if len(lm_list) != 0:
            # Tip of Index (8) and Middle (12) and Thumb (4)
            x1, y1 = lm_list[8]
            x2, y2 = lm_list[12]
            xt, yt = lm_list[4]

            # Check which fingers are up
            # (MediaPipe Landmarks: 8 is index tip, 6 is index pip)
            index_up = y1 < lm_list[6][1]
            middle_up = y2 < lm_list[10][1]

            # 1. MOVEMENT MODE: Only Index Finger is Up
            if index_up and not middle_up:
                # Convert Coordinates using Numpy Interp
                # This maps the inner rectangle to the full screen resolution
                x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, screen_width))
                y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, screen_height))

                # Smoothening logic
                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y

            # 2. LEFT CLICK: Index and Middle are both up and close
            if index_up and middle_up:
                dist = get_distance((x1, y1), (x2, y2))
                if dist < 40:
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.1) # Small delay to prevent click spam

            # 3. RIGHT CLICK: Thumb and Index Finger pinch
            dist_right = get_distance((x1, y1), (xt, yt))
            if dist_right < 40:
                cv2.circle(frame, (xt, yt), 15, (0, 0, 255), cv2.FILLED)
                pyautogui.rightClick()
                time.sleep(0.2)

    # Display FPS (Optional)
    cv2.imshow("AI Virtual Mouse", frame)

    # Exit on 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()