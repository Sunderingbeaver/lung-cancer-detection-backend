from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import SimpleITK as sitk
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import base64
import traceback
import gdown
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware\


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("D:/Research SHIT/Repo/lung-cancer-detection/v0.0.2b.pt")
MODEL_DRIVE_ID = "1han39oMGCXQ-2allaEgqdK9UQFMfmJuW"

DEBUG_DIR = "debug_uploads"
os.makedirs(DEBUG_DIR, exist_ok=True)

def download_model():
    if not MODEL_PATH.exists():
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, str(MODEL_PATH), quiet=False)

download_model()
model = YOLO(str(MODEL_PATH))

def convert_dcm_to_jpeg(image_array):
    if len(image_array.shape) == 2:  # Grayscale image
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image_array[0], cv2.COLOR_GRAY2RGB)  # Use first frame
    
    return Image.fromarray(img_rgb)


async def load_dicom(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
        file_content = await file.read()  # ✅ Await file.read()
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    ds = sitk.ReadImage(temp_file_path)
    img_array = sitk.GetArrayFromImage(ds)

    # Ensure image is single-frame (H, W) and normalize
    if len(img_array.shape) == 3:
        img_array = img_array[0]  # Take first frame if multi-frame

    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img_array = np.uint8(img_array)  # Convert to uint8
    os.remove(temp_file_path)
    return img_array


def save_debug_image(image, filename):
    path = os.path.join(DEBUG_DIR, filename)
    image.save(path, format="JPEG")
    return path

@app.post("/detect/")
async def detect_lung_cancer(file: UploadFile = File(...), confidence: float = Form(...)):
    try:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension in ["dcm"]:
            img_array = await load_dicom(file)
            img = convert_dcm_to_jpeg(img_array)
        else:
            img = Image.open(file.file).convert("RGB")
        
        #before_backend_path = save_debug_image(img, "beforebackend.jpg")
        results = model.predict(source=np.array(img), conf=confidence)
        
        annotated_img = np.array(img)
        for box in results[0].boxes.xyxy.numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        after_img = Image.fromarray(annotated_img)
        #after_backend_path = save_debug_image(after_img, "afterbackend.jpg")
        
        buffered = BytesIO()
        after_img.save(buffered, format="JPEG")
        base64_encoded = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse(content={
            "detections": results[0].boxes.xywh.numpy().tolist() if results and len(results[0].boxes) > 0 else [],
            "confidence_scores": results[0].boxes.conf.numpy().tolist() if results and len(results[0].boxes) > 0 else [],
            "image": base64_encoded
        })
    except Exception as e:
        error_details = traceback.format_exc()  # Get full error traceback
        print("ERROR:\n", error_details)  # Print full error
        return JSONResponse(content={"error": str(e), "details": error_details}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Lung Cancer Detection API is running!"}
