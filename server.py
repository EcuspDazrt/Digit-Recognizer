from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from flask import jsonify
import uvicorn
import main
import numpy as np
import io
from PIL import Image
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
def predict(payload: dict):
    data = payload['image']
    data = data.split(',')[1]
    image_bytes = base64.b64decode(data)

    img = Image.open(io.BytesIO(image_bytes))

    img_gray = img.convert('L')
    img_arr = np.array(img_gray)
    img_arr = img_arr / 255.0

    digit, confidence = main.test_rand(img_arr.reshape(-1, 1))
    return {'digit': digit, 'confidence': confidence}


uvicorn.run(app, host="0.0.0.0", port=8000)