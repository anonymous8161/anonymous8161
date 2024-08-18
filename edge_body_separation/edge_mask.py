import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_tongue_regions(image_path, edge_width_ratio=0.191):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return None, None, None, None

    height, width = img.shape[:2]

    diagonal = np.sqrt(height**2 + width**2)

    edge_width = int(diagonal * edge_width_ratio)

    edge_width = edge_width if edge_width % 2 == 1 else edge_width + 1

    print(f"Image size: {width}x{height}, Calculated edge_width: {edge_width}")

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

    return img, edge_mask, edge_region, body_region

img, edge_mask, edge_region, body_region = get_tongue_regions('29.jpg')

if img is not None:
    plt.figure(figsize=(20,5))

    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(142)
    plt.title('Edge Mask')
    plt.imshow(edge_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(143)
    plt.title('Edge Region')
    plt.imshow(cv2.cvtColor(edge_region, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(144)
    plt.title('Body Region')
    plt.imshow(cv2.cvtColor(body_region, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Failed to process the image.")