from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image
import io
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from db import models, schemas, db

app = FastAPI()

# CORS 설정
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 초기화 (테이블 생성)
models.Base.metadata.create_all(bind=db.engine)

# 모델 로드
try:
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile, db: Session = Depends(db.get_db)):
    try:
        contents = await file.read()

        # 이미지를 PIL 포맷으로 변환
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # 이미지 전처리
        inputs = processor(images=image_np, return_tensors="pt")

        # 모델을 통해 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)

        # 예측 결과 처리
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        results = []
        for i, (logit, probability) in enumerate(zip(logits[0], probabilities[0])):
            class_name = model.config.id2label[i]
            rounded_probability = round(probability.item(), 2)
            results.append({"class": class_name, "probability": rounded_probability})

        # 확률에 따라 내림차순으로 정렬
        results.sort(key=lambda x: x["probability"], reverse=True)

        # 결과 저장: happy 감정이 0.9 이상일 때만
        if results[0]["class"] == "happy" and results[0]["probability"] >= 0.9:
            new_smileage = models.Smileage(
                mileage=10,  # 예시로 마일리지 증가
                emotion=results[0]["class"],
                probability=results[0]["probability"]
            )
            db.add(new_smileage)
            db.commit()

        return JSONResponse(content={"predictions": results[:3]})
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
