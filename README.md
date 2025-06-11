# Violence and Emotion Detection from Video

A comprehensive deep learning-based video analytics pipeline that detects **violence** and **facial emotions** in real-time or from pre-recorded videos. Ideal for enhancing safety in surveillance systems, behavioral monitoring, and forensic video review tasks.

---

## 🧠 Key Features

* 🎥 **Violence Detection** using a lightweight MobileNet + LSTM architecture
* 🙂 **Emotion Detection** leveraging [DeepFace](https://github.com/serengil/deepface)
* ✅ Multi-face support in each frame
* ♻️ Fusion logic: Interpret both physical and emotional states
* 🌟 Built-in **MLflow** integration for tracking experiments
* 🔄 CLI-ready for direct command line usage

---

## 📁 Project Structure

```
violence-detection/
├── core/
│   ├── model.py                  # Model architecture (MobileNet + LSTM)
│   ├── violence_detector.py      # Inference script for violence detection
│   ├── emotions_detector.py      # Combined violence + emotion detector
│   └── models/
│       └── best_model_mobilenet.pt  # Trained model weights
│
├── test_vids/                    # Sample videos for testing
├── training/
│   └── mobilenet-lstm.ipynb      # Notebook for model training
├── Pipfile                       # Pipenv dependency manager
└── README.md                     # Project documentation
```

---

## 🔧 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/violence-emotion-detector.git
cd violence-detection
```

### Install Dependencies

#### Recommended: Using Pipenv

```bash
pipenv install
pipenv shell
```

#### Alternative: Using pip

```bash
pip install torch torchvision opencv-python deepface mlflow numpy
```

---

## 🧪 Usage Guide

### 🎮 1. Run Violence Detection

```bash
python core/violence_detector.py --video test_vids/your_video.mp4
```

### 🙂 2. Run Emotion + Violence Detection

```bash
python core/emotions_detector.py --video test_vids/your_video.mp4
```

This script will:

* Detect all faces in the frame
* Classify emotions with DeepFace
* Predict violence using the LSTM model
* Show real-time bounding boxes with labels

---

## 📊 Output Format Example

Each detected face will be logged per frame:

```text
[Frame 024] Person 1: 😠 angry | 🚨 violent
[Frame 024] Person 2: 🙂 neutral | ✅ non-violent
```

---

## 🏋️ Training the Model

Open the training notebook:

```bash
cd training/
jupyter notebook mobilenet-lstm.ipynb
```

You can customize:

* Frame sampling rate & preprocessing
* CNN backbone (e.g. MobileNet, ResNet)
* LSTM hidden size, sequence length
* Logging with MLflow

Trained models should be saved in:

```bash
core/models/best_model_mobilenet.pt
```

---

## 📊 MLflow Integration (Optional)

Track and visualize your experiment results:

```python
import mlflow
mlflow.start_run()
...
mlflow.log_metric("accuracy", acc)
mlflow.log_param("model_type", "mobilenet-lstm")
```

Start MLflow UI:

```bash
mlflow ui
```

---

## 🔮 Future Enhancements

* [ ] Add **audio classification** support for richer context
* [ ] Develop a **web dashboard** using FastAPI/Streamlit
* [ ] Implement a **Transformer-based fusion model**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Support & Contributions

If you found this project helpful:

* Star the repository on GitHub
* Fork and contribute via pull requests
* Reach out with collaboration proposals!
