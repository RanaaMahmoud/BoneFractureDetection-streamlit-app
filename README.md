# 🦴 Bone Fracture Detection with YOLOv8 and Streamlit

This project is an AI-powered bone fracture detection tool built using [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and Streamlit for the user interface. The goal is to assist medical professionals in identifying fractures from X-ray images accurately and efficiently.

## 🚀 Try It Out

👉 **Live Demo:** [Bone Fracture Detection Web App](https://bonefracturedetection-app-app-odmvyyqvucylax43q9vafp.streamlit.app/)

Upload an X-ray image, and the model will automatically detect and highlight the fractured area if present.

## 📁 Dataset

The model was trained using the **Fracture Multi-Region X-Ray Dataset**, which contains labeled X-ray images of various bone fractures.

🔗 **Dataset Source:** [Kaggle - Fracture Multi-Region X-Ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)

## 🧠 Model

We used **YOLOv8** (You Only Look Once v8), a state-of-the-art object detection model, fine-tuned on fracture X-ray images. The model can:

- Classify images as fractured or not
- Localize the fracture area with bounding boxes

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- Numpy, Pandas
- Ultralytics YOLOv8
- PyTorch
- Scikit-learn

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bonefracturedetection-streamlit-app.git
   cd bonefracturedetection-streamlit-app
