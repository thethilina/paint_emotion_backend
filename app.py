from fastapi import FastAPI, UploadFile, File
from services.image_emotion import predict_image_emotion

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ML server running"}

@app.post("/emotion/image")
async def image_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image_emotion(image_bytes)
    return result