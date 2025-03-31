import requests
import base64

url = "http://127.0.0.1:8000/detect/"  # Adjust if your FastAPI runs on a different port
files = {"file": open(r'd:\Research SHIT\Snall Nodules Testing\2.jpg', "rb")}  # Replace with a sample DICOM file
data = {"confidence": 0.5}
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    if "image" in result:
        with open("output.jpg", "wb") as f:
            f.write(base64.b64decode(result["image"]))
        print("Image saved as output.jpg. Check if it looks correct.")
    else:
        print("No image returned:", result)
else:
    print("Error:", response.text)
