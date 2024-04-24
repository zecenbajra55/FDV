from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import uvicorn
import PIL.Image
import PIL.ImageOps
import io
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO


labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'aa', 'ae', 'ah', 'ai', 'an', 'ana', 'au', 'ba', 'bha',
        'cha', 'chha', 'da', 'daa', 'dha', 'dhaa', 'ee', 'ga', 'gha', 'gya', 'ha', 'i', 'ja', 'jha', 'ka', 'kha', 'kna',
        'ksha', 'la', 'ma', 'motosaw', 'na', 'o', 'oo', 'pa', 'patalosaw', 'petchiryosaw', 'pha', 'ra', 'ta', 'taa', 'tha', 'thaa', 'tra', 'u', 'va', 'ya', 'yna']

# Load the trained model
model_path = "my_model.h5"  # Update with your model path
try:
    model = load_model(model_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Create FastAPI app instance
app = FastAPI()

# Define a route for making predictions
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Only images allowed"})
    
    # Read the image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Resize and preprocess the image
    resized_image = cv2.resize(image, (64, 64))
    normalized_image = resized_image / 255.0
    
    # Ensure input data is in the correct format
    input_data = np.expand_dims(normalized_image, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)  
    
    try:
        # Make prediction
        prediction = model.predict(input_data)
        predicted_class_index = int(np.argmax(prediction))  # Convert to regular Python integer
        predicted_class = labels[predicted_class_index]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error predicting: {str(e)}"})


@app.post("/display-resized-image/")
async def display_resized_image(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        return HTTPException(status_code=400, detail="Only images allowed")

    # Read the image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image
    resized_image = cv2.resize(image, (64, 64))

    # Encode the resized image to bytes
    _, img_encoded = cv2.imencode(".jpg", resized_image)
    img_bytes = img_encoded.tobytes()

    # Generate HTML to display the resized image
    html_content = """
    <html>
    <head>
    <title>Resized Image</title>
    </head>
    <body>
    <h2>Resized Image</h2>
    <img src="data:image/jpeg;base64,{}" />
    </body>
    </html>
    """.format(img_bytes.decode('utf-8'))

    # Return HTML response with the resized image
    return HTMLResponse(content=html_content, status_code=200)

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
