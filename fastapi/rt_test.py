from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import face_recognition
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64

# TODO: db에 연결
# known 이미지들이 있는 디렉토리 설정
known_faces_dir = "./images/known_images/"
known_face_encodings = []
known_face_names = []


# Known 얼굴들 로드 및 인코딩
def load_known_faces(directory):
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])


# Known 얼굴 로드
load_known_faces(known_faces_dir)

app = FastAPI()

# TODO: react에 연결
# Static 파일 서빙 (클라이언트 HTML 파일 제공)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 클라이언트로부터 이미지 데이터 수신
            image_data = await websocket.receive_bytes()
            pil_image = Image.open(io.BytesIO(image_data))
            image_to_compare = np.array(pil_image)

            # 얼굴 검출 및 인식
            face_locations = face_recognition.face_locations(image_to_compare)
            face_encodings = face_recognition.face_encodings(
                image_to_compare, face_locations
            )

            face_names = []
            face_distances_list = []
            if len(face_encodings) > 0:
                for face_encoding in face_encodings:

                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)

                    # Threshold를 설정하여 특정 거리 이상인 경우 "Unknown"으로 처리
                    if face_distances[best_match_index] < 0.4:  # 임계값 설정 (예: 0.6)
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"

                    face_names.append(name)
                    face_distances_list.append(face_distances[best_match_index])

                # 얼굴 주위에 box, name, distance 표시
                for (top, right, bottom, left), name, distance in zip(
                    face_locations, face_names, face_distances_list
                ):
                    cv2.rectangle(
                        image_to_compare, (left, top), (right, bottom), (0, 0, 255), 2
                    )
                    label = f"{name} ({distance:.2f})"
                    cv2.rectangle(
                        image_to_compare,
                        (left, bottom - 20),
                        (right, bottom),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(
                        image_to_compare,
                        label,
                        (left + 6, bottom - 6),
                        font,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # 얼굴 인식 여부에 관계없이 이미지를 Base64로 인코딩하여 클라이언트로 전송
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

    # except Exception as e:
    #     print(f"WebSocket error: {e}")
    #     await websocket.close()

    print(name)
