from fastapi import FastAPI, UploadFile, File

from services.image_emotion import predict_image_emotion


app = FastAPI()

# -----------------------
# IMAGE EMOTION ENDPOINT
# -----------------------
@app.post("/emotion/image")
async def image_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image_emotion(image_bytes)
    return result
