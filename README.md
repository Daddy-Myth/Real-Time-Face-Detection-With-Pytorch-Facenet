# Real-Time Face Detection with PyTorch FaceNet

Real-time face detection is an advanced application of computer vision that enables rapid and accurate identification of human faces from a video stream. This project utilizes MTCNN for face detection and FaceNet (InceptionResnetV1) for face recognition using PyTorch.

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

### 1. **MTCNN (Multi-task Cascaded Convolutional Networks)**

- **Purpose**: Face Detection
- **Functionality**:
  - Detects faces in images or webcam frames.
  - Identifies key facial landmarks (eyes, nose, mouth).
  - Crops and aligns face regions for further processing.
- **Why MTCNN?**:
  - Fast and accurate.
  - Works well under various lighting and face angles.
  - Ideal for real-time applications like webcam-based detection.

### 2. **FaceNet (InceptionResnetV1 Variant)**

- **Purpose**: Face Recognition / Feature Encoding
- **Functionality**:
  - Converts detected faces into 512-dimensional embedding vectors.
  - Each embedding represents the unique features of a face.
  - Enables face comparison using distance metrics (e.g., Euclidean distance).
- **Pretrained On**: [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) – a large-scale dataset with diverse facial images.
- **Why FaceNet?**:
  - High accuracy in face verification and recognition tasks.
  - Produces compact and robust embeddings.
  - Suitable for real-time performance and large-scale face matching.

## Directories 
```
Real-Time Face Detection/
├── RTFD.ipynb   # Folder with known face images
└── Saved        # With Pictures to be modeled
     ├── person1.jpg
     └── person2.jpg
```

## How It Works

### 1. Install Required Libraries

Install the dependencies with:

```bash
pip install torch facenet_pytorch opencv-python tqdm
```


### 2. Load Pretrained Models

We use two models from `facenet-pytorch`:

- **MTCNN**: Detects and crops faces from images or webcam frames.
- **InceptionResnetV1** (pretrained on VGGFace2): Converts faces into 512-dimensional embeddings.

```python
from facenet_pytorch import MTCNN, InceptionResnetV1
```


### 3. Prepare Saved Faces

- Create a folder named `./saved/`
- Add face images you want to recognize
- I added John Cena's and mine
- <img src="https://github.com/user-attachments/assets/1d60403a-9b09-478e-a760-e7dbcbb1590b" alt="image" width="500"/>


For each image:

- Load it with OpenCV
- Crop the face with MTCNN
- Generate a face embedding with InceptionResnet
- Store it in a dictionary for fast comparison

```python
all_people_faces[person_name] = encode(cropped_face)[0, :]
```

### 4. Live Webcam Detection

- Open the webcam using OpenCV.
- For each frame:
  - Detect and crop faces using `mtcnn.detect_box()`
  - Generate embeddings for each face
  - Compare them to the saved ones using Euclidean distance
  - Identify the closest match or label as `"Undetected"`

```python
distance = (known_embedding - live_embedding).norm().item()
```

### 5. Display Frame Rate (FPS)

- Track time between frames using `time.time()`
- Calculate and display real-time FPS using OpenCV

### 6. Running the System

Since this is in a Jupyter Notebook, just run the cells in order:

- Import libraries
- Load models
- Encode saved faces
- Run the `detect()` function to start real-time recognition
- Press `q` while the video window is selected to quit the webcam feed.

## Final Output in live webcam feed
<img src="https://github.com/user-attachments/assets/c07cb187-3d9f-4a22-ba9b-b326518e3ef0" alt="image" width="500"/>
