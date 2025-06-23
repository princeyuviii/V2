import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("❌ Failed to access webcam")
else:
    print("✅ Webcam working!")