import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# Initialize webcam
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Initial variables
c_dir = 'A'
save_dir = f"C:\\Users\\suhasini s hombal\\OneDrive\\Desktop\\Phase2 project major\\AtoZ_3.1\\{c_dir}\\"
count = len(os.listdir(save_dir))
offset = 15
step = 1
flag = False
suv = 0

# Create base white image
white_template_path = "C:\\Users\\suhasini s hombal\\OneDrive\\Desktop\\Phase2 project major\\white.jpg"
white_base = np.ones((400, 400), np.uint8) * 255
cv2.imwrite(white_template_path, white_base)

while True:
    try:
        success, frame = capture.read()
        if not success or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        hands, _ = hd.findHands(frame, draw=False)
        white = cv2.imread(white_template_path)

        skeleton1 = None  # Reset for each frame

        if hands:
            hand = hands[0]
            if isinstance(hand, dict) and 'bbox' in hand:
                x, y, w, h = hand['bbox']
                cropped = frame[max(0, y - offset): y + h + offset, max(0, x - offset): x + w + offset]

                handz, imz = hd2.findHands(cropped, draw=True, flipType=True)

                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    osx = ((400 - w) // 2) - 15
                    osy = ((400 - h) // 2) - 15

                    for t in range(0, 4):  # Thumb
                        cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy), (pts[t + 1][0] + osx, pts[t + 1][1] + osy), (0, 255, 0), 2)
                    for t in range(5, 8):  # Index
                        cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy), (pts[t + 1][0] + osx, pts[t + 1][1] + osy), (0, 255, 0), 2)
                    for t in range(9, 12):  # Middle
                        cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy), (pts[t + 1][0] + osx, pts[t + 1][1] + osy), (0, 255, 0), 2)
                    for t in range(13, 16):  # Ring
                        cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy), (pts[t + 1][0] + osx, pts[t + 1][1] + osy), (0, 255, 0), 2)
                    for t in range(17, 20):  # Pinky
                        cv2.line(white, (pts[t][0] + osx, pts[t][1] + osy), (pts[t + 1][0] + osx, pts[t + 1][1] + osy), (0, 255, 0), 2)

                    # Palm lines
                    cv2.line(white, (pts[5][0] + osx, pts[5][1] + osy), (pts[9][0] + osx, pts[9][1] + osy), (0, 255, 0), 2)
                    cv2.line(white, (pts[9][0] + osx, pts[9][1] + osy), (pts[13][0] + osx, pts[13][1] + osy), (0, 255, 0), 2)
                    cv2.line(white, (pts[13][0] + osx, pts[13][1] + osy), (pts[17][0] + osx, pts[17][1] + osy), (0, 255, 0), 2)
                    cv2.line(white, (pts[0][0] + osx, pts[0][1] + osy), (pts[5][0] + osx, pts[5][1] + osy), (0, 255, 0), 2)
                    cv2.line(white, (pts[0][0] + osx, pts[0][1] + osy), (pts[17][0] + osx, pts[17][1] + osy), (0, 255, 0), 2)

                    # Landmarks
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + osx, pts[i][1] + osy), 2, (0, 0, 255), 1)

                    skeleton1 = white.copy()

                    # Grayscale and binary views
                    gray = cv2.cvtColor(skeleton1, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

                    # Show windows
                    cv2.imshow("Skeleton", skeleton1)
                    cv2.imshow("Grayscale", gray)
                    cv2.imshow("Binary", binary)

        # Info on frame
        cv2.putText(frame, f"dir={c_dir} count={count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('n'):  # Switch to next letter
            c_dir = chr(ord(c_dir) + 1) if c_dir != 'Z' else 'A'
            save_dir = f"C:\\Users\\suhasini s hombal\\OneDrive\\Desktop\\Phase2 project major\\AtoZ_3.1\\{c_dir}\\"
            count = len(os.listdir(save_dir))
            print(f"Switched to: {c_dir}")
            flag = False
        elif key == ord('a'):  # Toggle capture flag
            flag = not flag
            suv = 0
            print("Flag set to:", flag)

        if flag and skeleton1 is not None:
            if suv == 180:
                flag = False
                print("Auto capture stopped.")
            if step % 3 == 0:
                filepath = os.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(filepath, skeleton1)
                count += 1
                suv += 1
            step += 1
        elif flag and skeleton1 is None:
            print("skeleton1 not ready yet...")

    except Exception:
        print("Error:", traceback.format_exc())

# Cleanup
capture.release()
cv2.destroyAllWindows()
