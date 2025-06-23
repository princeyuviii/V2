import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os

st.set_page_config(layout="wide")
st.title("👕 Virtual Shirt Try-On App")

shirt_folder = "shirts"
shirt_files = os.listdir(shirt_folder)
shirt_index = st.slider("Choose Shirt", 0, len(shirt_files)-1, 0)
shirt_path = os.path.join(shirt_folder, shirt_files[shirt_index])
shirt_image = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

frame_window = st.image([])

def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x + w > bw or y + h > bh or x < 0 or y < 0:
        return background

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            (1 - alpha) * background[y:y+h, x:x+w, c] +
            alpha * overlay[:, :, c]
        )
    return background

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]

        shoulder_width = int(abs(r_shoulder.x - l_shoulder.x) * frame.shape[1])
        shirt_width = shoulder_width + 60
        shirt_height = int(shirt_width * 1.2)

        center_x = int((l_shoulder.x + r_shoulder.x) / 2 * frame.shape[1]) - shirt_width // 2
        center_y = int(l_shoulder.y * frame.shape[0]) - 20

        shirt_resized = cv2.resize(shirt_image, (shirt_width, shirt_height))
        frame = overlay_transparent(frame, shirt_resized, center_x, center_y)

    frame_window.image(frame, channels="BGR")

cap.release()