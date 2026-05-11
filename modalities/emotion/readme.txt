Use mediapipe for face detection
Use mobilenet V2 as pretrained model
Dataset RAF-DB


Emotion Model Overview
The emotion model is a face emotion recognition system that classifies facial expressions into 7 emotions. Here's how it works:

Architecture
Base Model: MobileNetV2 (pretrained on ImageNet) - a lightweight CNN ideal for real-time inference
Output Layer: Modified to classify 7 emotions instead of 1000 ImageNet classes
Emotions: Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral

Training Pipeline
Dataset: RAF-DB (Reliable Affective Faces Database)
Loss Function: CrossEntropyLoss
Optimizer: Adam (learning rate: 0.0001)
Duration: 25 epochs with batch size 32
Input Size: 224×224 images with data augmentation

Real-time Inference
Face Detection: MediaPipe detects faces in video frames
Face Extraction: Crops detected face regions from the frame
Preprocessing: Resizes to 224×224 and applies normalization
Prediction: Passes to MobileNetV2, outputs softmax probabilities for each emotion
Output: Returns predicted emotion label with confidence percentage

Key Files
mobilenet_emotion.py - Model architecture definition
train.py - Training loop with validation
realtime_inference.py - Real-time emotion detection from webcam
config.py - Configuration (7 emotion classes, learning rate, epochs, etc.)