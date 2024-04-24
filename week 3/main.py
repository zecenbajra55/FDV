from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model_path = "my_model.h5"  # Replace with the path to your saved model
model = load_model(model_path)

# Create FastAPI app instance
app = FastAPI()

# Define a route for making predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess the image (resize and normalize)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    
    # Get the predicted class label
    predicted_class = np.argmax(prediction[0])
    
    return {"class_id": predicted_class}