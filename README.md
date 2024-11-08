# FaceMatch: Face Similarity Detection in Python

FaceMatch is a Python-based face similarity detection system that uses OpenCV's pre-trained models to find and compare faces across images. It’s designed to streamline facial recognition tasks by allowing quick identification of similar faces in image collections.

# Features

  -Load a reference image and compare it with multiple images to find similar faces.
  -Uses deep learning with OpenCV to extract facial embeddings.
  -Calculates cosine similarity to identify matching faces.
  -Customizable similarity threshold for flexible matching.

# Prerequisites

Python 3.6+<br/>
OpenCV<br/>
NumPy<br/>
Scikit-Learn<br/>

# Model Files

To use FaceMatch, you need the following model files, which should be saved in the project directory:

<b>deploy.prototxt</b>: Contains the model architecture.<br/>
<b>res10_300x300_ssd_iter_140000.caffemodel</b>: Contains the model weights.

# How It Works

<b>Face Detection</b>: The model first detects faces in each image by generating bounding boxes around them.<br/>
<b>Embedding Generation</b>: Each detected face is cropped and resized to a standard 96x96 format, then transformed into an embedding, a numerical representation of the face.<br/>
<b>Similarity Calculation</b>: Using cosine similarity, the embeddings are compared to determine how similar each face is to the reference face.<br/>

# Adjusting the Similarity Threshold

The similarity threshold <b>(default: 0.6)</b> is adjustable, depending on the level of accuracy and similarity required:

<b>Higher Threshold</b>: Increases matching strictness but may reduce the number of matched faces.<br/>
<b>Lower Threshold</b>: Relaxes the matching criteria, which can result in more potential matches but possibly more false positives.<br/>
