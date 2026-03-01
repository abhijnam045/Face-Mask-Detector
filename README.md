Face-Mask-Detector

Real-time face mask detection system using CNN and OpenCV for binary image classification.

Features

- Detects whether a person is wearing a mask
- Real-time detection using webcam
- CNN-based image classification
- Face detection using Haar Cascade

Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- CNN (Convolutional Neural Network)

Dataset
Binary dataset with two classes:
- With Mask
- Without Mask
  the sample dataset I took: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

How to Run

1. Install dependencies
pip install -r requirements.txt

2. Train model
python train_model.py

3. Run real-time detection
python detect_mask.py


Results
Achieved ~96% validation accuracy.

Author
Abhijna Medapati
