#FaceMatch: Face Similarity Detection in Python

FaceMatch is a Python-based face similarity detection system that uses OpenCV's pre-trained models to find and compare faces across images. It’s designed to streamline facial recognition tasks by allowing quick identification of similar faces in image collections.

# Features

  -Load a reference image and compare it with multiple images to find similar faces.
  -Uses deep learning with OpenCV to extract facial embeddings.
  -Calculates cosine similarity to identify matching faces.
  -Customizable similarity threshold for flexible matching.

# Prerequisites

Python 3.6+
OpenCV
NumPy
Scikit-Learn

# Model Files

To use FaceMatch, you need the following model files, which should be saved in the project directory:

<b>deploy.prototxt</b>: Contains the model architecture.
res10_300x300_ssd_iter_140000.caffemodel: Contains the model weights.

