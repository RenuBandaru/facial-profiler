# facial-profiler
Multi-Task Facial Profiler is a deep learning project that performs joint prediction of emotion, age, and gender from a single face image. The pipeline covers data preprocessing, multi-dataset fusion, model training, evaluation, and real-time inference via webcam.

# Facial Profiler: Real-Time Age, Gender, & Emotion Detection
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time facial analysis system that predicts **age bins (0-9 to 60+), gender (M/F), and emotion (7 classes)** using a multi-task EfficientNetB0 model trained on merged UTKFace + RAF-DB datasets.

## 🎯 Features
- **Multi-task learning**: Single model predicts age/gender/emotion simultaneously
- **Live webcam inference**: Real-time detection at 5-15 FPS (CPU)
- **Production-ready**: Model saved as portable `.keras` format
- **Cross-platform**: macOS/Windows (tested)

## 🏗️ Architecture

UTKFace (23.7K) + RAF-DB (12K) → Pseudo-labeling → Merged Dataset (36K)
↓
EfficientNetB0 (backbone, frozen) + 3 classification heads
↓
Custom masked loss → Training (80/20 split)
↓
Live inference: MTCNN → Crop → Predict → Visualize


**Key Innovation**: Handles missing labels with task-specific masks during training.

## 📊 Dataset & Results

### Merged Dataset Statistics

- Age Bins (7 classes): [0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+]

  - Distribution: 20-29 (12K) > 0-9 (4.8K) > 60+ (2.9K)

- Gender (2 classes): Male (20K) vs Female (16K)

- Emotions (7 classes): Happy (4.8K), Neutral (2.5K), Fear (2K), others


### Training Results (Validation)

 * Age Accuracy: 65-75% (best: 20-29 bin @ 85%)
 * Gender Accuracy: 89-92%
 * Emotion Accuracy: 68-72% (Happy/Neutral dominant)
 * Overall: Multi-task handles label sparsity well


## 🚀 Quick Start

#### 1. Clone & Environment
```bash
git clone https://github.com/YOUR_USERNAME/facial-profiler.git
cd facial-profiler
pip install -r requirements.txt
```
#### 2. Train Model
```bash
# Data prep (UTKFace + RAF-DB)
jupyter notebook pseudo_labeling.ipynb  # Creates merged_dataset.csv

# Train EfficientNet multi-task
jupyter notebook facial_recognition.ipynb  # Saves facial_multitask_model.keras
````
#### 3. Live Demo
```bash
python live_predict.py  # Webcam inference
```


🔧 Requirements

tensorflow>=2.15
opencv-python>=4.8
mtcnn>=0.1.1
numpy
pandas
matplotlib
jupyter
scikit-learn
tqdm

🎓 Methods Used

- Pseudo-labeling: Train demographics model → predict emotion-only images
- Multi-task EfficientNetB0: Shared backbone + 3 task heads (age/gender/emotion)
- Masked losses: Ignore missing labels (-1) per task
- Two-phase training: Frozen backbone → fine-tune top layers
- Real-time pipeline: MTCNN → EfficientNet → OpenCV visualization

📈 Results Summary

Validation Accuracy:
- Age: 65-75% (7 bins)
- Gender: 89-92% (binary)
- Emotion: 68-72% (7 classes)

Live FPS: 5-15 (CPU), 30+ (optimized)
Model Size: 16MB (.keras)


📝 Citation

Built with TensorFlow 2.20, EfficientNetB0 (Google AI), 
UTKFace Dataset, RAF-DB Basic Dataset

🤝 Contributing
PRs welcome! Focus: TFLite conversion, Android/iOS ports, dataset augmentation.

📄 License
MIT License - see LICENSE © 2026

