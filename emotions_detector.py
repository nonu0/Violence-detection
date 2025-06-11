import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import MobileNetLstmModel
from deepface import DeepFace
import torch.nn as nn

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetLstmModel()
CLASS_NAMES = ['NonViolence', 'Violence']

model.load_state_dict(torch.load(r"C:\Users\Administrator\work\mlflow_trial\core\models\best_model_mobilevnet_lstm.pt", map_location=device))
model.to(device)
model.eval()

r3d_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([232, 232], interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Emotion fusion logic
def fuse_predictions(violence_label, violence_conf, emotion_label):
    if violence_conf >= 0.7:
        return violence_label, violence_conf
    elif violence_conf >= 0.5 and emotion_label in ['angry', 'fear', 'disgust']:
        return "Violence", violence_conf + 0.15
    elif emotion_label in ['happy', 'neutral', 'surprise']:
        return "NonViolence", violence_conf - 0.1
    else:
        return violence_label, violence_conf

# Emotion detection using DeepFace
def get_emotion_from_frame(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception:
        return "unknown"

# Main prediction function
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames, 16 if total_frames >= 16 else total_frames, dtype=int, endpoint=False)
    frames = []

    # Track center frame for emotion analysis
    emotion_frame = None
    center_frame_idx = frame_idxs[len(frame_idxs) // 2]

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            transformed = r3d_transform(frame)
            frames.append(transformed)

            if idx == center_frame_idx:
                emotion_frame = frame.copy()
        else:
            frames.append(torch.zeros(3, 224, 224))

    cap.release()

    clip = torch.stack(frames).unsqueeze(0).to(device)  # Shape: [1, T, C, H, W]
    with torch.no_grad():
        output = model(clip)
        prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_class].item()
        violence_label = CLASS_NAMES[pred_class]

    # Emotion prediction
    emotion = get_emotion_from_frame(emotion_frame) if emotion_frame is not None else "unknown"
    print(f"Detected emotion: {emotion}")

    # Fusion
    final_label, fused_conf = fuse_predictions(violence_label, confidence, emotion)
    print(f"Violence model: {violence_label} ({confidence:.2f}) → Final: {final_label} ({fused_conf:.2f})")

    return final_label, fused_conf, emotion

# Display with overlays
def display_prediction(video_path):
    label, confidence, emotion = predict_video(video_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        overlay_text = f"{label} ({confidence:.2f}) | Emotion: {emotion}"
        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Prediction", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test it
video_path = r"C:\Users\Administrator\work\mlflow_trial\core\test_vids\5.webm"
display_prediction(video_path)
