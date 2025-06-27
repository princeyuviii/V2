import os
import numpy as np
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = PoseDetector()

# Load shirt images
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)

pantFolderPath = "Resources/Pants"
listPants = os.listdir(pantFolderPath)
pantNumber = 1


imageNumber = 0

# Load button images safely
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)

if imgButtonRight is None or imgButtonLeft is None:
    raise FileNotFoundError("❌ Could not load 'Resources/button.png'. Make sure the file exists and is a valid PNG with transparency.")

buttonH, buttonW = imgButtonRight.shape[:2]

counterRight = 0
counterLeft = 0
selectionSpeed = 10

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
    right_x_btn = min(frame_w - buttonW, int(frame_w * 0.9))
    left_x_btn = max(0, int(frame_w * 0.05))
    y_btn = min(frame_h - buttonH, int(frame_h * 0.4))

    required_indices = [11, 12, 23, 24, 25, 26, 27, 28]
    if all(idx < len(lmList) and lmList[idx][1] != 0 and lmList[idx][2] != 0 for idx in required_indices):
        lm11 = lmList[11][1:3]  # Left shoulder
        lm12 = lmList[12][1:3]  # Right shoulder
        lm23 = lmList[23][1:3]  # Left hip
        lm24 = lmList[24][1:3]  # Right hip
        lm25 = lmList[25][1:3]  # Left knee
        lm26 = lmList[26][1:3]  # Right knee
        lm27 = lmList[27][1:3]  # Left ankle
        lm28 = lmList[28][1:3]  # Right ankle

        # ---- SHIRT OVERLAY ----
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        top_y = int(min(lm11[1], lm12[1]))
        bottom_y = int(max(lm23[1], lm24[1]))
        left_x = int(min(lm11[0], lm12[0], lm23[0], lm24[0]))
        right_x = int(max(lm11[0], lm12[0], lm23[0], lm24[0]))

        shoulder_width = np.linalg.norm(np.array(lm11) - np.array(lm12))
        reference_width = 200  # or store first-time shoulder width
        scale_factor = shoulder_width / reference_width

        shirt_width = right_x - left_x
        shirt_height = bottom_y - top_y

        shirt_width = int(shirt_width * scale_factor)
        shirt_height = int(shirt_height * scale_factor)

        shirt_width = max(50, min(shirt_width, 600))
        shirt_height = max(50, min(shirt_height, 600))


        if shirt_width > 0 and shirt_height > 0:
            imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))
            try:
                frame_h, frame_w = img.shape[:2]

                if bottom_y > frame_h:
                    shirt_height = frame_h - top_y
                    imgShirt = imgShirt[:shirt_height, :, :]

                if right_x > frame_w:
                    shirt_width = frame_w - left_x
                    imgShirt = imgShirt[:, :shirt_width, :]

                img = cvzone.overlayPNG(img, imgShirt, (left_x, top_y))
            except Exception as e:
                print("⚠️ Shirt overlay failed:", e)
        else:
            print("⚠️ Invalid dimensions for shirt overlay.")

        # ---- PANT OVERLAY ----
        try:
            pantImg = cv2.imread(os.path.join(pantFolderPath, listPants[pantNumber]), cv2.IMREAD_UNCHANGED)
            if pantImg is None:
                print("❌ Pant image could not be loaded.")
            else:
                print("📦 Pant image loaded successfully.")

                hip_center = np.mean([lm23, lm24], axis=0).astype(int)
                ankle_center = np.mean([lm27, lm28], axis=0).astype(int)

                pant_height = int(np.linalg.norm(ankle_center - hip_center))
                pant_width = int(np.linalg.norm(np.array(lm23) - np.array(lm24))) + 40

                if pant_width > 0 and pant_height > 0:
                    pantImg = cv2.resize(pantImg, (pant_width, pant_height))

                    top_left_x = int(hip_center[0] - pant_width // 2)
                    top_left_y = int(hip_center[1])

                    frame_h, frame_w = img.shape[:2]
                    if top_left_y + pant_height > frame_h:
                        pantImg = pantImg[:frame_h - top_left_y, :, :]

                    if top_left_x + pant_width > frame_w:
                        pantImg = pantImg[:, :frame_w - top_left_x, :]

                    img = cvzone.overlayPNG(img, pantImg, (top_left_x, top_left_y))
                else:
                    print("⚠️ Invalid pant dimensions.")
        except Exception as e:
            print("⚠️ Pants overlay failed:", e)

    else:
        print("⚠️ Required landmarks not detected properly.")


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
