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
import asyncio
from typing import List, Dict, Any, Optional

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

# Ultralytics Inference API configuration
url = "https://predict.ultralytics.com"
headers = {"x-api-key": "9bd505fceee65248698a13aed81476344a034b0626"}

# Define the three model URLs
model_urls = [
    "https://hub.ultralytics.com/models/TrTYx7AMvwoC70Y13Kbo",  # v0.0.2C
    "https://hub.ultralytics.com/models/2g0rAQmsahbQIaLzaOs9",    # v0.0.2B
    "https://hub.ultralytics.com/models/jYCzxD59FYR7Aa5pRmCQ"      # v0.1.0
]

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

# Function to make an API request to a specific model
async def call_model_api(model_url: str, image_bytes: bytes, confidence: float) -> Dict[str, Any]:
    try:
        data = {
            "model": model_url,
            "imgsz": 640,
            "conf": confidence,
            "iou": 0.45
        }
        
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        
        # Make synchronous request in async function
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(url, headers=headers, data=data, files=files)
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"Error calling model {model_url}: {str(e)}")
        return {"error": str(e)}

# Process model results to get predictions
def extract_predictions(inference_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions = []
    if "images" in inference_results and len(inference_results["images"]) > 0:
        results = inference_results["images"][0].get("results", [])
        for result in results:
            prediction = {
                "xyxy": [
                    result["box"]["x1"],
                    result["box"]["y1"],
                    result["box"]["x2"],
                    result["box"]["y2"]
                ],
                "confidence": result["confidence"],
                "label": result["name"]
            }
            predictions.append(prediction)
    return predictions

# Get model with highest confidence
def get_highest_confidence_result(all_predictions: List[List[Dict[str, Any]]]) -> tuple:
    max_confidence = -1
    best_predictions = []
    best_model_index = -1
    
    for i, predictions in enumerate(all_predictions):
        if not predictions:
            continue
            
        # Find highest confidence in this model's predictions
        model_max_conf = max((p.get("confidence", 0) for p in predictions), default=0)
        
        if model_max_conf > max_confidence:
            max_confidence = model_max_conf
            best_predictions = predictions
            best_model_index = i
    
    return best_predictions, best_model_index

# Fallback for unsupported HTTP methods
@app.api_route("/detect/", methods=["GET", "PUT", "DELETE", "PATCH"])
async def invalid_method():
    return JSONResponse(status_code=405, content={"error": "Use POST with form-data"})

# Endpoint to detect lung cancer using multiple Ultralytics models
@app.post("/detect/")
async def detect_lung_cancer(file: UploadFile = File(...), confidence: float = Form(...)):
    try:
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "dcm":
            content = await file.read()
            await file.seek(0)
            img_array = await load_dicom(file)
            img = convert_dcm_to_jpeg(img_array)
        else:
            content = await file.read()
            await file.seek(0)
            img = Image.open(BytesIO(content)).convert("RGB")

        image_bytes = compress_image(img, quality=50)

        # Step 1: Call first model
        first_model_result = await call_model_api(model_urls[0], image_bytes, confidence)
        first_predictions = extract_predictions(first_model_result)
        filtered_first_predictions = [p for p in first_predictions if p.get("label") != "Healthy Lung"]

        if filtered_first_predictions:
            max_conf = max((p.get("confidence", 0) for p in filtered_first_predictions), default=0)
            if max_conf >= 0.7:
                print("Early return with Model 1 due to high confidence prediction.")
                annotated_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                for box in filtered_first_predictions:
                    x1, y1, x2, y2 = map(int, box["xyxy"])
                    label = box.get("label", "Unknown")
                    conf_score = box.get("confidence", 0)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_img, f"{label} {conf_score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                final_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                base64_encoded = base64.b64encode(compress_image(final_img, quality=50)).decode()

                return JSONResponse(content={
                    "detections": filtered_first_predictions,
                    "confidence_scores": [p.get("confidence", 0) for p in filtered_first_predictions],
                    "image": base64_encoded,
                    "model_used": 1
                })

        # Step 2: Call remaining models if confidence < 0.7
        other_model_results = await asyncio.gather(*[
            call_model_api(model_url, image_bytes, confidence)
            for model_url in model_urls[1:]
        ])

        all_predictions = [filtered_first_predictions]  # Include first model’s results
        for result in other_model_results:
            predictions = extract_predictions(result)
            filtered = [p for p in predictions if p.get("label") != "Healthy Lung"]
            all_predictions.append(filtered)

        best_predictions, best_model_index = get_highest_confidence_result(all_predictions)
        confidence_scores = [p.get("confidence", 0) for p in best_predictions]

        annotated_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for box in best_predictions:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            label = box.get("label", "Unknown")
            conf_score = box.get("confidence", 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, f"{label} {conf_score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        final_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        base64_encoded = base64.b64encode(compress_image(final_img, quality=50)).decode()

        return JSONResponse(content={
            "detections": best_predictions,
            "confidence_scores": confidence_scores,
            "image": base64_encoded,
            "model_used": best_model_index + 1 if best_model_index >= 0 else None
        })

    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=500, content={"error": "Inference API error", "details": str(e)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "details": traceback.format_exc()})


# Root endpoint to check if API is running
@app.get("/")
async def root():
    return {"message": "Multi-Model Lung Cancer Detection API is running!"}

#To run the FastAPI app locally comment out for deployment
#if __name__ == "__main__":
#   uvicorn.run(app, host="0.0.0.0", port=8000)
