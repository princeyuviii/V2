import os
import numpy as np
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = PoseDetector()

shirtFolderPath = "Resources/Shirts"
pantFolderPath = "Resources/Pants"
listShirts = sorted(os.listdir(shirtFolderPath))
listPants = sorted(os.listdir(pantFolderPath))

imageNumber = 0
pantNumber = 0

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Camera read failed.")
        continue

    img = cv2.flip(img, 1)
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    frame_h, frame_w = img.shape[:2]

    if lmList and len(lmList) > 28:
        lm11 = lmList[11][1:3]  # Left shoulder
        lm12 = lmList[12][1:3]  # Right shoulder
        lm23 = lmList[23][1:3]  # Left hip
        lm24 = lmList[24][1:3]  # Right hip
        lm27 = lmList[27][1:3]  # Left ankle
        lm28 = lmList[28][1:3]  # Right ankle

        # ---- SHIRT OVERLAY ----
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber % len(listShirts)]), cv2.IMREAD_UNCHANGED)

        top_y = int(min(lm11[1], lm12[1])) + 60  # camera offset
        bottom_y = int(max(lm23[1], lm24[1]))
        left_x = int(min(lm11[0], lm12[0], lm23[0], lm24[0]))
        right_x = int(max(lm11[0], lm12[0], lm23[0], lm24[0]))

        shirt_width = max(50, right_x - left_x)
        shirt_height = max(50, bottom_y - top_y)

        # Clamp to frame
        top_y = max(0, top_y)
        left_x = max(0, left_x)
        shirt_width = min(shirt_width, frame_w - left_x)
        shirt_height = min(shirt_height, frame_h - top_y)

        if imgShirt is not None:
            imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))
            try:
                img = cvzone.overlayPNG(img, imgShirt, (left_x, top_y))
            except Exception as e:
                print("⚠️ Shirt overlay failed:", e)

        # ---- PANT OVERLAY ----
        pantImg = cv2.imread(os.path.join(pantFolderPath, listPants[pantNumber % len(listPants)]), cv2.IMREAD_UNCHANGED)

        if pantImg is not None:
            hip_center = np.mean([lm23, lm24], axis=0).astype(int)
            ankle_center = np.mean([lm27, lm28], axis=0).astype(int)
            pant_height = int(np.linalg.norm(ankle_center - hip_center))
            pant_width = int(np.linalg.norm(np.array(lm23) - np.array(lm24))) + 40

            top_left_x = int(hip_center[0] - pant_width // 2)
            top_left_y = int(hip_center[1])

            # Clamp
            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            pant_width = min(pant_width, frame_w - top_left_x)
            pant_height = min(pant_height, frame_h - top_left_y)

            if pant_width > 0 and pant_height > 0:
                pantImg = cv2.resize(pantImg, (pant_width, pant_height))
                try:
                    img = cvzone.overlayPNG(img, pantImg, (top_left_x, top_left_y))
                except Exception as e:
                    print("⚠️ Pants overlay failed:", e)

    else:
        print("⚠️ Required landmarks not detected properly.")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key == ord('d'):
        imageNumber += 1
    elif key == ord('a'):
        imageNumber -= 1
    elif key == ord('w'):
        pantNumber += 1
    elif key == ord('s'):
        pantNumber -= 1

cap.release()
cv2.destroyAllWindows()
