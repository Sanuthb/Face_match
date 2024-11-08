import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'deploy.prototxt')
    weights_path = os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel')
    
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)
    return net

def get_face_embedding(image, model):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            return face_blob.flatten()
    return None

def find_similar_faces(reference_folder_path, upload_folder_path, model, threshold=0.6):
    ref_image_path = None
    for filename in os.listdir(reference_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            ref_image_path = os.path.join(reference_folder_path, filename)
            break

    if ref_image_path is None:
        print("No reference image found in the Reference folder.")
        return []

    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        print(f"Error: Could not read the reference image at {ref_image_path}")
        return []

    ref_embedding = get_face_embedding(ref_image, model)
    if ref_embedding is None:
        print("No face found in the reference image.")
        return []

    similar_images = []

    for filename in os.listdir(upload_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(upload_folder_path, filename)
            group_image = cv2.imread(image_path)
            if group_image is None:
                print(f"Error: Could not read the image at {image_path}")
                continue

            group_embedding = get_face_embedding(group_image, model)
            if group_embedding is not None:
                similarity = cosine_similarity([ref_embedding], [group_embedding])[0][0]
                if similarity >= threshold:
                    similar_images.append((image_path, similarity))

    similar_images = sorted(similar_images, key=lambda x: x[1], reverse=True)
    return similar_images

reference_folder_path = os.path.join(os.path.dirname(__file__), '../uploads/Reference')
upload_folder_path = os.path.join(os.path.dirname(__file__), '../uploads')
model = load_model()
similar_images = find_similar_faces(reference_folder_path, upload_folder_path, model, threshold=0.6)

similar_image_names = [os.path.basename(img_path[0]) for img_path in similar_images]

print(json.dumps(similar_image_names))
