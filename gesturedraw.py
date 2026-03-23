import cv2
import mediapipe as mp
import numpy as np
import time
import os

# -------------------- MediaPipe Setup --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------- Camera Setup --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# -------------------- Canvas --------------------
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
history = []

prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0
last_clear = 0

# -------------------- Drawing Settings --------------------
draw_color = (255, 0, 255)   # Purple
brush_thickness = 6
eraser_size = 30

# -------------------- Finger Detection --------------------
def fingers_up(hand):
    tips = [8, 12, 16, 20]
    fingers = []
    for tip in tips:
        fingers.append(hand.landmark[tip].y < hand.landmark[tip - 2].y)
    return fingers, sum(fingers)

# -------------------- Toolbar --------------------
def draw_toolbar(frame):
    colors = [(0,0,255),(0,255,0),(255,0,0),(255,0,255)]
    x = 20
    for c in colors:
        cv2.rectangle(frame, (x,20), (x+60,80), c, -1)
        x += 80

    cv2.putText(frame, f"Thickness: {brush_thickness}",
                (900,60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,255), 2)

# -------------------- FPS --------------------
prev_time = 0

# -------------------- Full Screen Window --------------------
cv2.namedWindow("GestureDraw PRO", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "GestureDraw PRO",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# -------------------- Main Loop --------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_toolbar(frame)

    # FPS calculation
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}",
                (1150,40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w, _ = frame.shape

            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            # Gesture smoothing
            smooth_x = int(0.7 * smooth_x + 0.3 * x)
            smooth_y = int(0.7 * smooth_y + 0.3 * y)

            fingers, count = fingers_up(hand_landmarks)

            # 🎨 Color Selection (Index finger on toolbar)
            if count == 1 and y < 90:
                if 20 < x < 80:
                    draw_color = (0,0,255)
                elif 100 < x < 160:
                    draw_color = (0,255,0)
                elif 180 < x < 240:
                    draw_color = (255,0,0)
                elif 260 < x < 320:
                    draw_color = (255,0,255)

            # ✏ Drawing
            if count == 1:
                if prev_x == 0:
                    prev_x, prev_y = smooth_x, smooth_y
                    history.append(canvas.copy())
                cv2.line(canvas,
                         (prev_x, prev_y),
                         (smooth_x, smooth_y),
                         draw_color, brush_thickness)
                prev_x, prev_y = smooth_x, smooth_y

            # 🧽 Eraser (fist)
            elif count == 0:
                cv2.circle(canvas,
                           (smooth_x, smooth_y),
                           eraser_size, (0,0,0), -1)
                prev_x, prev_y = 0, 0

            # 🧹 Clear Canvas (4 fingers – hold)
            elif count == 4:
                if time.time() - last_clear > 1:
                    history.append(canvas.copy())
                    canvas = np.zeros((720,1280,3), dtype=np.uint8)
                    last_clear = time.time()

            else:
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Merge canvas
    frame = cv2.add(frame, canvas)

    # Footer instructions
    cv2.putText(frame,
        "1 Finger: Draw | 0: Erase | 4: Clear | S: Save | Z: Undo | Q: Exit",
        (30,700), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255,255,255), 2)

    cv2.imshow("GestureDraw PRO", frame)

    key = cv2.waitKey(1) & 0xFF

    # 💾 Save drawing
    if key == ord('s'):
        if not os.path.exists("drawings"):
            os.makedirs("drawings")
        filename = f"drawings/drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

    # 🔙 Undo
    if key == ord('z') and history:
        canvas = history.pop()

    # ❌ Exit
    if key == ord('q'):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
