# Face Recognition App

This application uses OpenCV and DeepFace to recognize known faces in real-time through a webcam feed.

## Features

- Load known faces from a specified directory.
- Real-time face detection and recognition.
- Displays recognized names or "Unknown" for unrecognized faces.
- Uses the Facenet model for embedding generation.

## Requirements

- Python 3.x
- OpenCV
- DeepFace
- NumPy

## Installation

1. Clone the repository or download the source code.
2. Install the required packages:

   ```
   pip install opencv-python deepface numpy
   ```

3. Prepare a directory named `known_faces` containing images of known individuals (in `.jpg`, `.png`, or `.jpeg` formats).

4. Run the application:

   ```
   python app.py
   ```

5. Allow access to the webcam when prompted.

## Usage

- The application will start the webcam feed.
- It will detect and recognize faces in real-time.
- Press `q` to exit the application.

## Troubleshooting

- Ensure that your webcam is properly connected and accessible.
- Make sure the images in `known_faces` are clear and well-lit for better recognition results.
