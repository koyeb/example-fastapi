from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter
from catboost import CatBoostClassifier 

from predict import create_sentence,calculate_angles,calculate_pose_angles 

import numpy as np
import mediapipe as mp
import cv2
import asyncio

# python -m uvicorn main:app --reload --host 0.0.0.0 --port 9091

# source venv/bin/activate
# uvicorn main:app --reload --host 0.0.0.0 --port 9091

app = FastAPI()

origins = ["http://localhost:9090",
           "http://localhost:9091"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictions = []
predicted_mapping = {'child' : '아이',
                     'down' : '쓰러지다',
                     'lost' : '잃어버리다',
                     'report' : '신고하다',
                     'sick' : '아프다',
                     'toilet' : '화장실',
                     'wallet' : '지갑',
                     'where' : '어디'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

pre_model = CatBoostClassifier()
pre_model.load_model('cb_model.cbm')
        
@app.websocket("/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)

    temporary_predictions = [] #임시
    
    if not cap.isOpened():
        print("Cannot open camera")
        await websocket.close()
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # 이미지 처리
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        pose_result = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        arm_angles = np.zeros(2)
        landmark_data = np.zeros(18)
        right_hand_angles = np.zeros(15)
        right_hand_coords = np.zeros(63)
        left_hand_angles = np.zeros(15)
        left_hand_coords = np.zeros(63)   
        
        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                if i>= 2:
                    break
                
                hand_type = result.multi_handedness[i].classification[0].label
                angles, joint_coords = calculate_angles(hand_landmarks, img.shape)

                if hand_type == 'Right':
                    right_hand_angles = angles
                    right_hand_coords = joint_coords
                elif hand_type == 'Left':
                    left_hand_angles = angles
                    left_hand_coords = joint_coords   
        
        if pose_result.pose_landmarks:
            arm_angles = calculate_pose_angles(pose_result.pose_landmarks, img.shape)
            mp.solutions.drawing_utils.draw_landmarks(
            img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmark_data = []
            for idx in [11, 13, 15, 12, 14, 16]:
                landmark = pose_result.pose_landmarks.landmark[idx]
                landmark_data.extend([landmark.x, landmark.y, landmark.z])
            if len(landmark_data) == 0:
                landmark_data = np.zeros(18)
                
        data = np.concatenate((right_hand_angles, right_hand_coords, left_hand_angles, left_hand_coords, arm_angles, landmark_data))

        if websocket.client_state.value == 3:  # WebSocketState.CLOSED 상태
            break

        data = data.reshape(1,-1)
        predicted_label = pre_model.predict(data)
        
        # 예측된 라벨을 화면에 표시
        cv2.putText(img, text=str(predicted_label[0][0]), org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0,0), thickness=2)
        temporary_predictions.append(predicted_label[0][0])

        if len(temporary_predictions) >= 11:  # 2초동안 처리하는 이미지 수
            frequency = Counter(temporary_predictions)
            most_common_string = frequency.most_common(1)[0][0]
            temporary_predictions = []
            predictions.append(most_common_string)

        # 손 랜드마크 그리기
        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        
        _, buffer = cv2.imencode('.jpg', img)
        await websocket.send_bytes(buffer.tobytes())

        # 프레임 속도 조절을 위한 대기
        await asyncio.sleep(0.1)
  
@app.get("/stop")
def get_predictions():
    
    predicted_text = []
    for word in predictions:
        predicted_text.append(predicted_mapping.get(word))

    trans_sentence = create_sentence(predicted_text)
    predictions.clear()
    
    return JSONResponse(content={"text":trans_sentence})
