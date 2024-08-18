import os
import cv2
import numpy as np

def resize_and_pad(img, target_size):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_h / h, target_w / w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded

def add_text_bars(img, top_text, bottom_text, green_border_thickness=2):
    h, w = img.shape[:2]
    bar_height = 80
    
    top_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    cv2.putText(top_bar, top_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    bottom_bar = np.full((bar_height, w, 3), (255, 255, 255), dtype=np.uint8)  # 
    # cv2.rectangle(bottom_bar, (0, 0), (w-1, bar_height-1), (0, 255, 0), green_border_thickness)  # green
    cv2.rectangle(bottom_bar, (0, 0), (w-1, bar_height-1), (0, 0, 255), green_border_thickness)  # red
    cv2.putText(bottom_bar, bottom_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
    
    return np.vstack((top_bar, img, bottom_bar))

def concat_images(folder1, folder2, folder3, output_folder, green_border_thickness=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(folder1) if f.endswith('.png')]

    max_height = 0
    max_width = 0

    for file in files:
        img1_path = os.path.join(folder1, file)
        img2_path = os.path.join(folder2, f"original_points_cropped_{file}")
        img3_path = os.path.join(folder3, f"points_cropped_{file}")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img3 = cv2.imread(img3_path)

        if img1 is None or img2 is None or img3 is None:
            continue

        max_height = max(max_height, img2.shape[0])
        max_width += max(img1.shape[1], img2.shape[1], img3.shape[1])

    target_height = max_height
    target_width = max_width // len(files)

    for idx, file in enumerate(files):
        img1_path = os.path.join(folder1, file)
        img2_path = os.path.join(folder2, f"original_points_cropped_{file}")
        img3_path = os.path.join(folder3, f"points_cropped_{file}")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img3 = cv2.imread(img3_path)

        if img1 is None or img2 is None or img3 is None:
            print(f"Error: Unable to read one or more images for {file}")
            continue

        img1 = resize_and_pad(img1, (target_height, target_width // 3))
        img2 = cv2.resize(img2, (target_width // 3, target_height))
        img3 = cv2.resize(img3, (target_width // 3, target_height))

        combined_img = np.hstack((img1, img2, img3))
        
        top_text = "GT [Pale 0, TipSideRed 1, Spot 0, Black 0, Crack 1; Toothmark 1; FurThick 1; FurYellow 0]"  # GT 
        bottom_text = "Ours [Pale 0, TipSideRed 1, Spot 0, Black 0; Crack 1; Toothmark 1; FurThick 1; FurYellow 0]"  # Ours
        combined_img_with_text = add_text_bars(combined_img, top_text, bottom_text, green_border_thickness)

        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, combined_img_with_text)

        print(f"Processed: {file}")

    print("All images have been processed.")


folder1 = "final_mosaic"
folder2 = "seg_cropped_images_original_with_points"
folder3 = "seg_cropped_images_align_with_points"
output_folder = "./combined_images"

concat_images(folder1, folder2, folder3, output_folder, green_border_thickness=10)