import os
import cv2
from alignment_V2 import align_segmented_tongue

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    successful_count = 0
    total_count = 0

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                total_count += 1
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path, file)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    color_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
                    if color_image is None:
                        print(f"Failed to load image: {input_path}")
                        continue

                    aligned_image, _, _, _, _, _, _ = align_segmented_tongue(color_image)
                    cropped_aligned_image = aligned_image
                    cv2.imwrite(output_path, cropped_aligned_image)
                    
                    successful_count += 1
                    if successful_count % 100 == 0:
                        print(f"Processed {successful_count} images successfully out of {total_count} total images.")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

    print(f"Processing completed. Total images: {total_count}, Successfully processed: {successful_count}")

if __name__ == "__main__":
    input_folder = "seg_cropped_images"  # 替换为您的输入文件夹路径
    output_folder = "seg_cropped_images_align"  # 替换为您想要保存处理后图像的文件夹路径
    
    process_images(input_folder, output_folder)