import cv2
import torch
import numpy as np
from torchvision import transforms
import timm
from PIL import Image
from collections import deque, Counter

# --- 配置 ---
MODEL_PATH = 'mobilevit_emotion_recognition.pth'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

# 为每个情绪定义不同的颜色 (BGR格式)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),   # 红色
    'Fear': (0, 165, 255), # 橙色
    'Happy': (0, 255, 0),   # 绿色
    'Sad': (255, 0, 0),     # 蓝色
    'Suprise': (0, 255, 255) # 黄色
}

# --- 平滑处理参数 ---
BOX_SMOOTHING_FRAMES = 5
PREDICTION_SMOOTHING_FRAMES = 15 # A larger buffer for emotion stability
box_history = deque(maxlen=BOX_SMOOTHING_FRAMES)
emotion_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
NO_FACE_THRESHOLD = 15
no_face_frames = 0
last_smoothed_emotion = None

# --- 初始化 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 加载模型 ---
try:
    model = timm.create_model('mobilevit_s', pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# --- 加载人脸检测器 ---
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    print(f"无法加载Haar级联分类器文件: {HAAR_CASCADE_PATH}")
    exit()
print("人脸检测器加载成功！")

# --- 打开摄像头 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头。")
    exit()
print("摄像头已启动。按 'q' 键退出。")

# --- 主循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 1:
        # --- 多人脸模式 (会闪烁) ---
        box_history.clear()
        emotion_history.clear()
        no_face_frames = 0
        last_smoothed_emotion = None
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                input_tensor = data_transform(face_pil)
                input_batch = input_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_batch)
                
                emotion = CLASSES[torch.argmax(output, 1).item()]
                color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                text = f"{emotion}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.9, 2)
                cv2.rectangle(frame, (x, y + h), (x + text_width, y + h + text_height + baseline), color, -1)
                cv2.putText(frame, text, (x, y + h + text_height), font, 0.9, (0, 0, 0), 2)

    elif len(faces) == 1:
        # --- 单人脸模式 (平滑处理) ---
        no_face_frames = 0
        box_history.append(faces[0])
        avg_box = np.mean(box_history, axis=0).astype(int)
        ax, ay, aw, ah = avg_box

        face_roi = frame[ay:ay+ah, ax:ax+aw]
        if face_roi.size > 0:
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            input_tensor = data_transform(face_pil)
            input_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch)
            
            raw_emotion = CLASSES[torch.argmax(output, 1).item()]
            emotion_history.append(raw_emotion)
            
            smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]
            last_smoothed_emotion = smoothed_emotion
            
            color = EMOTION_COLORS.get(smoothed_emotion, (0, 255, 0))
            
            cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), color, 2)
            
            text = f"{smoothed_emotion}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.9, 2)
            cv2.rectangle(frame, (ax, ay + ah), (ax + text_width, ay + ah + text_height + baseline), color, -1)
            cv2.putText(frame, text, (ax, ay + ah + text_height), font, 0.9, (0, 0, 0), 2)

    else:
        # --- 无人脸模式 ---
        no_face_frames += 1
        if no_face_frames > NO_FACE_THRESHOLD:
            box_history.clear()
            emotion_history.clear()
            last_smoothed_emotion = None
        
        if len(box_history) > 0 and last_smoothed_emotion:
            avg_box = np.mean(box_history, axis=0).astype(int)
            ax, ay, aw, ah = avg_box
            color = EMOTION_COLORS.get(last_smoothed_emotion, (0, 255, 0))
            cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), color, 2)
            text = f"{last_smoothed_emotion}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.9, 2)
            cv2.rectangle(frame, (ax, ay + ah), (ax + text_width, ay + ah + text_height + baseline), color, -1)
            cv2.putText(frame, text, (ax, ay + ah + text_height), font, 0.9, (0, 0, 0), 2)

    cv2.imshow('Real-time Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("程序已退出。")