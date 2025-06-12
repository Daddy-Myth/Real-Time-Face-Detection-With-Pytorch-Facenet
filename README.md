# Real-Time Face Detection with PyTorch FaceNet

Real-time face detection is an advanced application of computer vision that enables rapid and accurate identification of human faces from a video stream. This project utilizes MTCNN for face detection and FaceNet (InceptionResnetV1) for face recognition using PyTorch.

## Demo
Coming soon 

## Features
- Real-time face detection using webcam
- Face recognition with FaceNet embeddings
- Label prediction with distance thresholding
- Auto-labeling of "Undetected" faces
- Jupyter-friendly and also script-ready (main.py)
- FPS (Frames Per Second) overlay to monitor performance

## Libraries Used
- Pytorch - Deep learning framework
- facenet-pytorch – Pretrained MTCNN + InceptionResnetV1
- OpenCV – Webcam capture and image display
- tqdm – Progress bars for data processing

## Models Used
- MTCNN (Multi-task Cascaded Convolutional Networks):
Used for detecting face bounding boxes and facial landmarks.

- FaceNet (InceptionResnetV1):
Used for encoding facial features into 512-dimensional embeddings for comparison.

## Learnings
