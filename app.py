from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

import numpy as np
import cv2

from detector import predict


app = FastAPI()


@app.get("/")
def home():

    return FileResponse("index.html")


@app.post("/predict")
async def spoof_predict(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(
        contents,
        np.uint8
    )

    image = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )

    result = predict(image)

    return result