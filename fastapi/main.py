from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

import face_recognition
from typing import List
from utils.fr_modules import compare_faces_with_similarity


# 이미지 업로드 함수
def load_image_from_upload(upload_file: UploadFile):
    image = Image.open(io.BytesIO(upload_file.file.read()))
    return np.array(image)


app = FastAPI()

origins = [
    "http://localhost:3000",  # React 애플리케이션의 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 출처 리스트
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 모델 로드
try:
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile):
    try:
        contents = await file.read()

        # 이미지를 PIL 포맷으로 변환
        image = Image.open(io.BytesIO(contents))

        # RGB로 변환
        image = image.convert("RGB")

        # NumPy 배열로 변환
        image_np = np.array(image)

        # 이미지 전처리
        inputs = processor(images=image_np, return_tensors="pt")

        # 모델을 통해 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)

        # logits에서 예측 클래스와 확률 계산
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # 결과를 저장
        results = []
        for i, (logit, probability) in enumerate(zip(logits[0], probabilities[0])):
            class_name = model.config.id2label[i]
            rounded_probability = round(probability.item(), 2)
            results.append({"class": class_name, "probability": rounded_probability})

        # 확률에 따라 내림차순으로 정렬
        results.sort(key=lambda x: x["probability"], reverse=True)

        return JSONResponse(content={"predictions": results[:3]})
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# compare 추가
@app.post("/compare-faces/")
async def compare_faces(known_image_file: UploadFile, unknown_image_file: UploadFile):
    # Load the images from the uploaded files
    known_image = load_image_from_upload(known_image_file)
    unknown_image = load_image_from_upload(unknown_image_file)

    # Encode the faces
    try:
        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    except IndexError:
        raise HTTPException(
            status_code=400, detail="No face found in one of the images."
        )

    # Compare faces
    tolerance = 0.6
    results = face_recognition.compare_faces(
        [known_encoding], unknown_encoding, tolerance=tolerance
    )
    # 얼굴 비교 및 유사도 계산
    similarity = compare_faces_with_similarity([known_encoding], unknown_encoding)

    # Convert numpy.bool_ to standard bool (fastapi가 numpy.bool_를 fastapi가 처리 못함)
    return {
        "match": bool(results[0]),
        "tolerance": tolerance,
        "similarity": similarity[0][1],
    }
