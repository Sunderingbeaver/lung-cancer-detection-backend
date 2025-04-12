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

# Allow all origins (for CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            "model": model_url,  # URL to your model on the Hub
            "imgsz": 640,        # Image size
            "conf": confidence,  # Confidence threshold
            "iou": 0.45          # IoU threshold for non-max suppression
        }

        # Make the inference request to the Ultralytics API
        response = requests.post(url, headers=headers, data=data, files={"file": image_bytes})

        # Check for successful response
        response.raise_for_status()

        # Parse the JSON response from Ultralytics
        inference_results = response.json()

        # Assuming that the 'predictions' field contains the bounding boxes and related data
        predictions = inference_results[0].get('predictions', []) if inference_results else []

        # Annotate the image with the bounding boxes returned by the API
        annotated_img = np.array(img)
        for box in predictions:
            x1, y1, x2, y2 = box['xyxy']  # Adjust as per the actual structure of each box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Save annotated image in memory
        after_img = Image.fromarray(annotated_img)

        # Compress the image (quality set to 50 for example)
        base64_encoded = base64.b64encode(compress_image(after_img, quality=50)).decode()

        # Return JSON response with detection results and annotated image
        return JSONResponse(content={
            "detections": predictions,  # List of detections
            "confidence_scores": [box['confidence'] for box in predictions],
            "image": base64_encoded  # Base64-encoded image with annotations
        })
    
    except requests.exceptions.RequestException as e:
        # Handle request error
        return JSONResponse(status_code=500, content={"error": "Failed to communicate with inference API", "details": str(e)})
    except Exception as e:
        # Handle other exceptions
        error_details = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "details": error_details})

# Root endpoint to check if API is running
@app.get("/")
async def root():
    return {"message": "Lung Cancer Detection API is running!"}

# Optional: To run the FastAPI app locally
#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
