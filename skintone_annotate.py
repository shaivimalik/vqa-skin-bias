import os
import cv2
import dlib
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Skin tone annotation using Monk Scale")
parser.add_argument("--img_dir", required=True, help="Directory containing images")
parser.add_argument("--metadata_csv", required=True, help="Path to metadata CSV file")
parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
parser.add_argument("--orb_dir", required=True, help="Directory containing Monk scale reference images")
parser.add_argument("--landmarks_model", default="shape_predictor_68_face_landmarks.dat", 
                    help="Path to dlib landmarks model")
args = parser.parse_args()

# Load Monk scale reference colors from orb directory
monk_scale = {}
for orb_path in os.listdir(args.orb_dir):
    # Load and convert reference image to RGB
    orb = cv2.imread(os.path.join(args.orb_dir, orb_path))
    orb = cv2.cvtColor(orb, cv2.COLOR_BGR2RGB)

    # Flatten image to get all pixels and remove duplicates
    orb_pixels = orb.reshape(-1, 3)
    orb_pixels = np.unique(orb_pixels, axis=0)

    monk_scale['Monk'+orb_path.split('.')[0][3:]] = orb_pixels

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to extract region and exclude lips
def extract_region(image, landmarks, indices, remove_lips=False):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw main region (e.g., cheeks or chin)
    points = landmarks[indices]
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

    # exclude lips (outer and inner)
    if remove_lips:
        outer_lips = landmarks[48:60]  # Outer lip
        cv2.fillConvexPoly(mask, cv2.convexHull(outer_lips), 0)

    # Apply mask
    region = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(points)
    roi = region[y:y+h, x:x+w]
    return roi, mask

# Process each image in the directory
result = []
for img_path in os.listdir(args.img_dir):
    # Load image and convert to grayscale for face detection
    image = cv2.imread(os.path.join(args.img_dir, img_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = detector(gray)

    # Skip images without exactly one face
    if len(faces)!=1: 
        result.append(None)
        continue

    # Get facial landmarks for the detected face
    face = faces[0]
    shape = predictor(gray, face)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    # Define cheek region indices (jawline + nose bridge + part of mouth area)
    cheek_indices = list(range(1, 16)) + list(range(31, 36)) + list(range(48, 55)) 

    cheek, cheek_mask = extract_region(image, landmarks, cheek_indices, remove_lips=True)

    # Convert to RGB and flatten for pixel-wise processing
    masked_img = cv2.cvtColor(cheek, cv2.COLOR_BGR2RGB)
    masked_img = masked_img.reshape(-1, 3)

    # Find closest Monk scale tone for each pixel
    output_map = np.empty(masked_img.shape[0], dtype=object)

    for idx, pixel in enumerate(masked_img):
        # Skip black pixels (masked out areas)
        if np.sum(pixel.astype(np.float32))==0: continue

        # Find closest Monk scale tone
        closest = None
        min_dist = float('inf')
        for tone, tone_pixels in monk_scale.items():
            diff = tone_pixels.astype(np.float32) - pixel.astype(np.float32)
            dists = np.sqrt(np.sum(diff**2, axis=1)) #np.linalg.norm(diff, axis=1) --> slower
            dist = np.min(dists)

            if dist < min_dist:
                min_dist = dist
                closest = tone

        output_map[idx] = closest

    # Determine most common tone
    unique, counts = np.unique(output_map, return_counts=True)
    result.append(unique[np.argmax(counts)])

# Load metadata and add results
df = pd.read_csv(args.metadata_csv)
df['monk_tone'] = result 
df.to_csv(args.output_csv) 