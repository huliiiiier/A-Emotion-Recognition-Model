from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
import numpy as np
from torchvision import transforms
import timm
from PIL import Image
import base64
import io
from waitress import serve

app = Flask(__name__, template_folder='.')
CORS(app)

# --- 配置 ---
MODEL_PATH = 'mobilevit_emotion_recognition.pth'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']
EMOTION_COLORS = {
    'Angry': '#FF0000', 'Fear': '#FFA500', 'Happy': '#00FF00',
    'Sad': '#0000FF', 'Suprise': '#FFFF00'
}

# --- 初始化 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 加载模型和检测器 ---
try:
    model = timm.create_model('mobilevit_s', pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载成功！")
    
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"无法加载Haar级联分类器: {HAAR_CASCADE_PATH}")
    print("人脸检测器加载成功！")
except Exception as e:
    print(f"初始化失败: {e}")
    model = None
    face_cascade = None

def predict_emotion(face_roi):
    """为单个人脸ROI预测情绪"""
    if not model: return None
    try:
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        input_tensor = data_transform(face_pil)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)
        return CLASSES[torch.argmax(output, 1).item()]
    except Exception:
        return None

def process_frame_for_emotions(image_data):
    """
    处理单个图像帧以进行情绪识别, 优先保证速度.
    - 不进行任何平滑处理.
    - 积极缩小图像尺寸以加快处理速度.
    """
    if not face_cascade: return {'error': '人脸检测器未加载'}

    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return {'error': '无法解码图像'}

        # --- 速度优化: 积极缩小图像 ---
        original_height, original_width = frame.shape[:2]
        scale = 1.0
        # 将图像最大宽度限制在320px以实现极速处理
        if original_width > 320:
            scale = 320 / original_width
            new_height = int(original_height * scale)
            frame = cv2.resize(frame, (320, new_height))
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- 速度优化: 使用更快的检测参数 ---
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=3, 
            minSize=(30, 30)
        )
        
        results = {'faces': []}
        
        # 对检测到的每个人脸进行处理
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                emotion = predict_emotion(face_roi)
                if emotion:
                    # 将坐标转换回原始尺寸
                    results['faces'].append({
                        'x': int(x / scale), 'y': int(y / scale),
                        'w': int(w / scale), 'h': int(h / scale),
                        'emotion': emotion,
                        'color': EMOTION_COLORS.get(emotion, '#00FF00')
                    })
        return results
        
    except Exception as e:
        print(f"处理帧时出错: {e}")
        return {'error': str(e)}

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': '请求缺少图像数据'}), 400
        
        # 直接调用新的、快速的处理函数
        result = process_frame_for_emotions(data['image'])
        return jsonify(result)
        
    except Exception as e:
        if 'ClientDisconnected' in str(type(e)):
            return jsonify({'error': '客户端断开连接'}), 204
        print(f"预测时发生未知错误: {e}")
        return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    print("服务器正在启动，请访问 http://127.0.0.1:8080")
    serve(app, host='0.0.0.0', port=8080)
