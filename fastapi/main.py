from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from PIL import Image
import cv2
import io
import os
import base64
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

import face_recognition
from typing import List
from utils.fr_modules import compare_faces_with_similarity
import mysql.connector
from mysql.connector import Error

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

# MySQL 연결 설정
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',  # 또는 '127.0.0.1'
            port=3306,          # 기본 MySQL 포트
            database='smileage',
            user='smileage',
            password='1234'
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None
    
# 이미지 업로드 함수
def load_image_from_upload(upload_file: UploadFile):
    image = Image.open(io.BytesIO(upload_file.file.read()))
    return np.array(image)

def compare_faces_with_similarity(known_encodings, unknown_encoding):
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    similarities = 1 - distances
    return list(zip(known_encodings, similarities))

# 얼굴 인코딩 데이터 로드
def load_registered_users_from_db():
    connection = create_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = connection.cursor()
    cursor.execute("SELECT userName, userFace FROM users")
    registered_users = cursor.fetchall()
    cursor.close()
    connection.close()
    
    known_face_encodings = []
    known_face_names = []
    
    for userName, userFace in registered_users:
        face_encoding = np.frombuffer(userFace, dtype=np.float64)
        known_face_encodings.append(face_encoding)
        known_face_names.append(userName)
    
    return known_face_encodings, known_face_names
@app.post("/verify-user")
async def verify_user(file: UploadFile = File(...)):
    try:
        # 파일 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # 얼굴 인코딩
        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        face_encoding = face_encodings[0]
        
        # DB에서 등록된 사용자 로드
        known_face_encodings, known_face_names = load_registered_users_from_db()
        
        # 사용자 인식
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < 0.4:
            name = known_face_names[best_match_index]
            return JSONResponse(content={"recognized": True, "userName": name})
        else:
            return JSONResponse(content={"recognized": False})
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    known_face_encodings, known_face_names = load_registered_users_from_db()

    try:
        while True:
            # 클라이언트로부터 이미지 데이터 수신
            image_data = await websocket.receive_bytes()
            pil_image = Image.open(io.BytesIO(image_data))
            image_to_compare = np.array(pil_image)

            # 얼굴 검출 및 인식
            face_locations = face_recognition.face_locations(image_to_compare)
            face_encodings = face_recognition.face_encodings(image_to_compare, face_locations)

            if face_encodings:
                for face_encoding in face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # 특정 거리 이하일 경우 등록된 사용자로 인식
                    if face_distances[best_match_index] < 0.4:
                        name = known_face_names[best_match_index]
                        # 등록된 사용자가 인식되었을 때 main.js로 넘어가는 처리
                        await websocket.send_text("User recognized: " + name)
                        await websocket.send_text("Redirect to main.js")
                        break
                    else:
                        await websocket.send_text("No registered user detected")

            # 얼굴 인식 여부와 상관없이 이미지를 Base64로 인코딩하여 클라이언트로 전송
            image_to_compare = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", image_to_compare)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(jpg_as_text)

    except WebSocketDisconnect:
        print("WebSocket disconnected unexpectedly")

    except Exception as e:
        print(f"WebSocket error: {e}")

    finally:
        if not websocket.client_state == WebSocketDisconnect:
            try:
                await websocket.close()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")


# 사용자 등록 엔드포인트
@app.post("/register-user")
async def register_user(file: UploadFile = File(...), userName: str = Form(...)):
    try:
        # 파일 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 얼굴 인코딩
        face_encodings = face_recognition.face_encodings(np.array(image))
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        face_encoding = face_encodings[0]
        
        # DB 연결
        connection = create_connection()
        if connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor()
        
        # 기존 유저의 얼굴 데이터 가져오기
        cursor.execute("SELECT userFace FROM users")
        registered_users = cursor.fetchall()

        # 등록된 유저들과 얼굴 비교
        for (registered_face,) in registered_users:
            registered_face_encoding = np.frombuffer(registered_face, dtype=np.float64)
            results = face_recognition.compare_faces([registered_face_encoding], face_encoding, tolerance=0.6)
            similarity = compare_faces_with_similarity([registered_face_encoding], face_encoding)[0][1]

            if results[0] and similarity >= 0.6:
                cursor.close()
                connection.close()
                return JSONResponse(content={"message": "User already registered", "proceed": True})

        # 새로운 유저 등록
        query = "INSERT INTO users (userName, userFace) VALUES (%s, %s)"
        cursor.execute(query, (userName, face_encoding.tobytes()))
        connection.commit()

        cursor.close()
        connection.close()

        return JSONResponse(content={"message": "User registered successfully", "proceed": False})
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
        inputs = processor(images=image_np, return_tensors="pt", padding=True)

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
    tolerance = 0.4
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
