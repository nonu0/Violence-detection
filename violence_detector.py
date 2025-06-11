import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from collections import deque
from model import MobileNetLstmModel
import torch.nn.functional as F

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_WINDOW = 16  # Number of frames per clip
IMG_SIZE = 224
CLASS_NAMES = ['NonViolence', 'Violence']
VIDEO_SOURCE = r'C:\Users\Administrator\work\mlflow_trial\core\test_vids\5.webm' # or 0 for webcam

# LOAD MODEL
model = MobileNetLstmModel()
model.load_state_dict(torch.load(
    r"C:\Users\Administrator\work\mlflow_trial\core\models\best_model_mobilevnet_lstm.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# TRANSFORMS
r3d_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([232, 232], interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# BUFFER FOR SLIDING WINDOW
frame_buffer = deque(maxlen=FRAME_WINDOW)


# INFERENCE LOOP
def real_time_inference():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    prediction_label = "Waiting..."
    confidence = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        frame_buffer.append(frame)

        if len(frame_buffer) == FRAME_WINDOW:
            # Preprocess frames
            processed_frames = [r3d_transform(f).unsqueeze(0) for f in frame_buffer]
            clip = torch.cat(processed_frames, dim=0).unsqueeze(0).to(DEVICE)  # Shape[1, T, C, H, W]

            with torch.no_grad():
                output = model(clip)
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()
                prediction_label = f"{CLASS_NAMES[pred_class]}"

        # Draw overlay
        overlay_text = f"{prediction_label} ({confidence:.2f})"
        color = (0, 0, 255) if prediction_label == "Violence" else (0, 255, 0)

        cv2.putText(display_frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Real-Time Violence Detection", display_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_inference()
