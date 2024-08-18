import cv2
import numpy as np
import os
from tqdm import tqdm

def get_tongue_regions(img, edge_width=30):
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tongue_contour = max(contours, key=cv2.contourArea)

    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.drawContours(full_mask, [tongue_contour], 0, 255, -1)

    kernel = np.ones((edge_width, edge_width), np.uint8)

    eroded = cv2.erode(full_mask, kernel, iterations=1)

    edge_mask = full_mask - eroded

    edge_mask[:height//5, :] = 0

    body_mask = full_mask.copy()
    body_mask[edge_mask == 255] = 0

    edge_mask_3channel = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR) / 255.0
    body_mask_3channel = cv2.cvtColor(body_mask, cv2.COLOR_GRAY2BGR) / 255.0

    edge_region = (img * edge_mask_3channel).astype(np.uint8)
    body_region = (img * body_mask_3channel).astype(np.uint8)

    return edge_region, body_region

def process_dataset(input_folder, edge_output_folder, body_output_folder):

    os.makedirs(edge_output_folder, exist_ok=True)
    os.makedirs(body_output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith('.jpg'):

                input_path = os.path.join(root, file)

                img = cv2.imread(input_path)
                if img is None:
                    print(f"Error: Could not read the image {input_path}")
                    continue

                edge_region, body_region = get_tongue_regions(img)

                rel_path = os.path.relpath(root, input_folder)

                edge_output_path = os.path.join(edge_output_folder, rel_path, file)
                body_output_path = os.path.join(body_output_folder, rel_path, file)

                os.makedirs(os.path.dirname(edge_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(body_output_path), exist_ok=True)

                cv2.imwrite(edge_output_path, edge_region)
                cv2.imwrite(body_output_path, body_region)

input_folder = 'dataset_align'
edge_output_folder = 'dataset_align_edge'
body_output_folder = 'dataset_align_body'

process_dataset(input_folder, edge_output_folder, body_output_folder)