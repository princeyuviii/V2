from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import os
from cvzone.PoseModule import PoseDetector
import cvzone
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

# Allow all frontend origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pose Detector
detector = PoseDetector()

shirtFolderPath = "Resources/Shirts"
pantFolderPath = "Resources/Pants"
listShirts = sorted(os.listdir(shirtFolderPath))
listPants = sorted(os.listdir(pantFolderPath))

def fetch_overlay_image(url):
    try:
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content)).convert("RGBA")
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    except:
        return None

def decode_base64_image(base64_string):
    header, encoded = base64_string.split(",", 1)
    img_data = base64.b64decode(encoded)
    img_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_img}"

@app.post("/tryon")
async def tryon(request: Request):
    data = await request.json()

    shirt_url = data.get("shirtUrl")
    pant_url = data.get("pantUrl")

    base64_img = data["image"]
    img = decode_base64_image(base64_img)
    frame_h, frame_w = img.shape[:2]

    img = cv2.flip(img, 1)
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, bboxWithHands=False, draw=False)

    shirtImg = fetch_overlay_image(shirt_url) if shirt_url else cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED)
    pantImg = fetch_overlay_image(pant_url) if pant_url else cv2.imread(os.path.join(pantFolderPath, listPants[0]), cv2.IMREAD_UNCHANGED)

    if lmList and len(lmList) > 28:
        lm11 = lmList[11][1:3]  # Left shoulder
        lm12 = lmList[12][1:3]  # Right shoulder
        lm23 = lmList[23][1:3]  # Left hip
        lm24 = lmList[24][1:3]  # Right hip
        lm27 = lmList[27][1:3]  # Left ankle
        lm28 = lmList[28][1:3]  # Right ankle

        # ---- SHIRT ----
        top_y = int(min(lm11[1], lm12[1])) + 60
        bottom_y = int(max(lm23[1], lm24[1]))
        left_x = int(min(lm11[0], lm12[0], lm23[0], lm24[0]))
        right_x = int(max(lm11[0], lm12[0], lm23[0], lm24[0]))

        shirt_width = max(50, right_x - left_x)
        shirt_height = max(50, bottom_y - top_y)
        top_y = max(0, top_y)
        left_x = max(0, left_x)
        shirt_width = min(shirt_width, frame_w - left_x)
        shirt_height = min(shirt_height, frame_h - top_y)

        if shirtImg is not None:
            shirtResized = cv2.resize(shirtImg, (shirt_width, shirt_height))
            try:
                img = cvzone.overlayPNG(img, shirtResized, (left_x, top_y))
            except:
                pass

        # ---- PANT ----
        hip_center = np.mean([lm23, lm24], axis=0).astype(int)
        ankle_center = np.mean([lm27, lm28], axis=0).astype(int)
        pant_height = int(np.linalg.norm(ankle_center - hip_center))
        pant_width = int(np.linalg.norm(np.array(lm23) - np.array(lm24))) + 40

        top_left_x = int(hip_center[0] - pant_width // 2)
        top_left_y = int(hip_center[1])

        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        pant_width = min(pant_width, frame_w - top_left_x)
        pant_height = min(pant_height, frame_h - top_left_y)

        if pantImg is not None and pant_width > 0 and pant_height > 0:
            pantResized = cv2.resize(pantImg, (pant_width, pant_height))
            try:
                img = cvzone.overlayPNG(img, pantResized, (top_left_x, top_left_y))
            except:
                pass

    processed = encode_image_to_base64(img)
    return {"processedImage": processed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")