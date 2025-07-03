import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Paths and assets
shirtFolderPath = "Resources/shirts"
pantFolderPath = "Resources/pants"
listShirts = sorted(os.listdir(shirtFolderPath))
listPants = sorted(os.listdir(pantFolderPath))
imageNumber = 0

imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
if imgButtonRight is None:
    print("Error loading right button image.")
    exit()
imgButtonLeft = cv2.flip(imgButtonRight, 1)

counterRight = 0
counterLeft = 0
selectionSpeed = 10

def overlay_image_alpha(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]
    x = max(0, min(x, background_width - 1))
    y = max(0, min(y, background_height - 1))
    h, w = overlay.shape[0], overlay.shape[1]
    if x + w > background_width:
        w = background_width - x
    if y + h > background_height:
        h = background_height - y
    if w <= 0 or h <= 0:
        return background
    overlay = cv2.resize(overlay, (w, h))
    if overlay.shape[2] < 4:
        overlay = np.concatenate([overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255], axis=2)
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
    return background

def adjust_pant_width(pant_img, desired_width):
    if pant_img is None or pant_img.shape[1] == 0:
        return pant_img
    height, width = pant_img.shape[:2]
    aspect_ratio = height / width
    new_height = int(desired_width * aspect_ratio)
    resized = cv2.resize(pant_img, (desired_width, new_height))
    return resized

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ Camera frame not available.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)
    ih, iw, _ = image.shape

    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        # SHIRT OVERLAY
        lm11_px = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * iw), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * ih))
        lm12_px = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * iw), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * ih))
        shirt_width = int(abs(lm11_px[0] - lm12_px[0]) * 1.5)
        shirt_height = int(shirt_width * 1.3)
        shirt_top_left = (
            max(0, min(iw - shirt_width, min(lm11_px[0], lm12_px[0]) - int(shirt_width * 0.15))),
            max(0, min(ih - shirt_height, min(lm11_px[1], lm12_px[1]) - int(shirt_height * 0.2)))
        )
        shirt_path = os.path.join(shirtFolderPath, listShirts[imageNumber])
        imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        if imgShirt is not None:
            imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))
            image = overlay_image_alpha(image, imgShirt, shirt_top_left[0], shirt_top_left[1])

        # PANT OVERLAY
        lm23_px = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * iw), int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * ih))
        lm24_px = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * iw), int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * ih))
        lm27_px = (int(lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * iw), int(lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * ih))
        lm28_px = (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * iw), int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * ih))
        pant_width = int(abs(lm23_px[0] - lm24_px[0]) * 1.3)
        pant_top_y = min(lm23_px[1], lm24_px[1])
        pant_bottom_y = max(lm27_px[1], lm28_px[1])
        pant_height = int(pant_bottom_y - pant_top_y)
        pant_top_left = (
            max(0, min(iw - pant_width, int((lm23_px[0] + lm24_px[0]) / 2 - pant_width / 2))),
            max(0, min(ih - pant_height, pant_top_y))
        )
        if listPants:
            pant_path = os.path.join(pantFolderPath, listPants[0])
            pantImg = cv2.imread(pant_path, cv2.IMREAD_UNCHANGED)
            if pantImg is not None and pant_width > 0 and pant_height > 0:
                pantImg = cv2.resize(pantImg, (int(pantImg.shape[1] * (pant_height / pantImg.shape[0])), pant_height))
                pantImg = adjust_pant_width(pantImg, pant_width)
                image = overlay_image_alpha(image, pantImg, pant_top_left[0], pant_top_left[1])

        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Overlay buttons
    image = overlay_image_alpha(image, imgButtonRight, image.shape[1] - imgButtonRight.shape[1] - 10, image.shape[0] // 2 - imgButtonRight.shape[0] // 2)
    image = overlay_image_alpha(image, imgButtonLeft, 10, image.shape[0] // 2 - imgButtonLeft.shape[0] // 2)

    # Hand gesture control for shirt switching
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * iw), int(tip.y * ih)

            if x > iw - imgButtonRight.shape[1] - 10 and image.shape[0] // 2 - 66 < y < image.shape[0] // 2 + 66:
                counterRight += 1
                cv2.ellipse(image, (iw - imgButtonRight.shape[1] // 2 - 10, image.shape[0] // 2), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    imageNumber = (imageNumber + 1) % len(listShirts)
            elif x < imgButtonLeft.shape[1] + 10 and image.shape[0] // 2 - 66 < y < image.shape[0] // 2 + 66:
                counterLeft += 1
                cv2.ellipse(image, (imgButtonLeft.shape[1] // 2 + 10, image.shape[0] // 2), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    imageNumber = (imageNumber - 1) % len(listShirts)
            else:
                counterRight = 0
                counterLeft = 0

    cv2.imshow("Virtual Try-On", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
