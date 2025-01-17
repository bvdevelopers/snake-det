from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

class FrameData(BaseModel):
    image: str

def preprocess_image(image_data):
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(img_data))
    img = img.resize((640, 640))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

@app.post("/detect_snake")
async def detect_snake(frame: FrameData):
    try:
        input_data = preprocess_image(frame.image)
        
        # Set up the input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the result
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process model output
        confidence_threshold = 0.5
        snake_detected = np.any(output_data[0][4] > confidence_threshold)

        if snake_detected:
            return {"snakeDetected": True}
        return {"snakeDetected": False}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
