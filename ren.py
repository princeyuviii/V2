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

# Path to shirt images and load list of shirts
shirtFolderPath = "Resources/shirts"
listShirts = os.listdir(shirtFolderPath)
print("List of shirts:", listShirts)
print(f"Number of shirts found: {len(listShirts)}")

# Shirt sizing parameters
fixedRatio = 262 / 190
shirtRatioHeightWidth = 591 / 490

imageNumber = 0

# Load button images
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
if imgButtonRight is None:
    print("Error loading right button image. Please check the path.")
    exit()
imgButtonLeft = cv2.flip(imgButtonRight, 1)

counterRight = 0
counterLeft = 0
selectionSpeed = 10

def overlay_image_alpha(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    # Ensure x and y are within bounds
    x = max(0, min(x, background_width - 1))
    y = max(0, min(y, background_height - 1))

    h, w = overlay.shape[0], overlay.shape[1]

    # Ensure the overlay fits within the background
    if x + w > background_width:
        w = background_width - x
    if y + h > background_height:
        h = background_height - y

    if w <= 0 or h <= 0:
        return background  # Nothing to overlay

    # Resize overlay if necessary
    overlay = cv2.resize(overlay, (w, h))

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find poses and hands
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)
    
    if pose_results.pose_landmarks:
        # Get landmarks for left and right shoulders
        lm11 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lm12 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Convert normalized coordinates to pixel coordinates
        ih, iw, _ = image.shape
        lm11_px = (int(lm11.x * iw), int(lm11.y * ih))
        lm12_px = (int(lm12.x * iw), int(lm12.y * ih))
        
        # Calculate shirt dimensions and position
        shirt_width = int(abs(lm11_px[0] - lm12_px[0]) * fixedRatio)
        shirt_height = int(shirt_width * shirtRatioHeightWidth)
        shirt_top_left = (
            max(0, min(iw - shirt_width, min(lm11_px[0], lm12_px[0]) - int(shirt_width * 0.15))),
            max(0, min(ih - shirt_height, min(lm11_px[1], lm12_px[1]) - int(shirt_height * 0.2)))  # Adjusted y-coordinate
        )
        
        # Load and resize shirt image
        imgShirtPath = os.path.join(shirtFolderPath, listShirts[imageNumber])
        imgShirt = cv2.imread(imgShirtPath, cv2.IMREAD_UNCHANGED)
        if imgShirt is not None:
            imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))
            
            # Overlay shirt on image
            image = overlay_image_alpha(image, imgShirt, shirt_top_left[0], shirt_top_left[1])
        
        # Draw pose landmarks for debugging
        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Overlay buttons
    image = overlay_image_alpha(image, imgButtonRight, image.shape[1] - imgButtonRight.shape[1] - 10, 
                                image.shape[0] // 2 - imgButtonRight.shape[0] // 2)
    image = overlay_image_alpha(image, imgButtonLeft, 10, image.shape[0] // 2 - imgButtonLeft.shape[0] // 2)
    
    # Check for button press using hand landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])

            # Right button detection
            if x > image.shape[1] - imgButtonRight.shape[1] - 10 and y > image.shape[0] // 2 - imgButtonRight.shape[0] // 2 and y < image.shape[0] // 2 + imgButtonRight.shape[0] // 2:
                counterRight += 1
                cv2.ellipse(image, (image.shape[1] - imgButtonRight.shape[1] // 2 - 10, image.shape[0] // 2), 
                            (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    imageNumber = (imageNumber + 1) % len(listShirts)
                    print(f"Switched to next shirt: {imageNumber}")
            
            # Left button detection
            elif x < imgButtonLeft.shape[1] + 10 and y > image.shape[0] // 2 - imgButtonLeft.shape[0] // 2 and y < image.shape[0] // 2 + imgButtonLeft.shape[0] // 2:
                counterLeft += 1
                cv2.ellipse(image, (imgButtonLeft.shape[1] // 2 + 10, image.shape[0] // 2), 
                            (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    imageNumber = (imageNumber - 1) if imageNumber > 0 else len(listShirts) - 1
                    print(f"Switched to previous shirt: {imageNumber}")
            
            else:
                counterRight = 0
                counterLeft = 0

    cv2.imshow('Virtual Try-On', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()