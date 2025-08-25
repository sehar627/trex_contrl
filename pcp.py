import mediapipe as mp
import pyautogui
import cv2

capture = cv2.VideoCapture(0)

width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands(max_num_hands=1)

# Only one key: "Jump"
key = "Jump"
key_size = 100
key_x, key_y = 350, 100   # fixed to match draw_button()

pinch = False

def draw_button(frame):
    cv2.rectangle(frame, (350, 100), (450, 150), (0, 0, 255), 3)
    cv2.putText(frame, "JUMP", (360, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def process_hand(image, hand_landmarks):
    global pinch

    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark

        # index finger tip
        finger_tip_x = int((landmarks[8].x) * width)
        finger_tip_y = int((landmarks[8].y) * height)

        # thumb tip
        thumb_tip_x = int((landmarks[4].x) * width)
        thumb_tip_y = int((landmarks[4].y) * height)

        # Draw pointer
        cv2.circle(image, (finger_tip_x, finger_tip_y), 8, (0, 255, 0), -1)

        # Distance between finger and thumb
        distance = ((finger_tip_x - thumb_tip_x) ** 2 + (finger_tip_y - thumb_tip_y) ** 2) ** 0.5

        # Check if finger is inside Jump button area
        if key_x < finger_tip_x < key_x + key_size and key_y < finger_tip_y < key_y + key_size:
            if distance < 20:   # pinch threshold
                if not pinch:
                    pinch = True
                    pyautogui.press("space")  # simulate space bar
            else:
                pinch = False
        else:
            pinch = False


while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(rgb_frame)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    draw_button(frame)
    process_hand(frame, hand_landmarks)

    cv2.imshow("On-Screen Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
