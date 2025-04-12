from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import SimpleITK as sitk
from io import BytesIO
from PIL import Image
import requests
import traceback
import base64
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app initialization
app = FastAPI()

'''
allowed_origins = [
    "http://localhost",  # Localhost testing
    "http://127.0.0.1", # For Postman (127.0.0.1)
    "https://testing-git-main-sunderingbeavers-projects.vercel.app/"  # Testing site
]'''

# Allow all origins (for CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ultralytics Inference API URL and API Key
url = "https://predict.ultralytics.com"
headers = {"x-api-key": "9bd505fceee65248698a13aed81476344a034b0626"}
model_url = "https://hub.ultralytics.com/models/TrTYx7AMvwoC70Y13Kbo"  # Replace with your model URL

# Convert DICOM to JPEG
def convert_dcm_to_jpeg(image_array):
    if len(image_array.shape) == 2:  # Grayscale image
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image_array[0], cv2.COLOR_GRAY2RGB)  # Use first frame
    return Image.fromarray(img_rgb)

# Load DICOM file
async def load_dicom(file):
    img_array = await file.read()
    img_array = np.frombuffer(img_array, dtype=np.uint8)
    ds = sitk.ReadImage(img_array)
    img_array = sitk.GetArrayFromImage(ds)

    # Ensure image is single-frame (H, W) and normalize
    if len(img_array.shape) == 3:
        img_array = img_array[0]  # Take first frame if multi-frame

    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img_array = np.uint8(img_array)  # Convert to uint8
    return img_array

# Function to compress the image by lowering the quality
def compress_image(image: Image, quality: int = 30):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)  # Lower the quality
    buffered.seek(0)  # Rewind the buffer
    return buffered.getvalue()  # Return the raw image data

# Fallback for unsupported HTTP methods
@app.api_route("/detect/", methods=["GET", "PUT", "DELETE", "PATCH"])
async def invalid_method():
    return JSONResponse(status_code=405, content={"error": "Use POST with form-data"})

# Endpoint to detect lung cancer using Ultralytics model
@app.post("/detect/")
async def detect_lung_cancer(file: UploadFile = File(...), confidence: float = Form(...)):
    try:
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension in ["dcm"]:
            img_array = await load_dicom(file)
            img = convert_dcm_to_jpeg(img_array)
        else:
            img = Image.open(file.file).convert("RGB")
        
        # Convert image to byte stream (required for sending to Ultralytics)
        image_bytes = compress_image(img, quality=50)

        # Prepare the inference data for Ultralytics API
        data = {
            "model": model_url,
            "imgsz": 640,
            "conf": confidence,
            "iou": 0.45
        }

        # Make the inference request to the Ultralytics API
        response = requests.post(url, headers=headers, data=data, files={"file": image_bytes})
        response.raise_for_status()

        inference_results = response.json()
        print("Ultralytics API response:", inference_results)

        # Make sure inference_results is a list and access the first item
        predictions = inference_results[0].get("predictions", []) if isinstance(inference_results, list) and inference_results else []

        # Convert PIL to OpenCV (BGR)
        annotated_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for box in predictions:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            conf = float(box.get("confidence", 0))
            label = box.get("label", "object")
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Convert back to PIL for encoding
        after_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

        # Encode the image with annotations
        base64_encoded = base64.b64encode(compress_image(after_img, quality=50)).decode()

        return JSONResponse(content={
            "detections": predictions,
            "confidence_scores": [box["confidence"] for box in predictions],
            "image": base64_encoded
        })

    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=500, content={"error": "Failed to communicate with inference API", "details": str(e)})
    except Exception as e:
        error_details = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "details": error_details})

# Root endpoint to check if API is running
@app.get("/")
async def root():
    return {"message": "Lung Cancer Detection API is running!"}

# Optional: To run the FastAPI app locally
#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
