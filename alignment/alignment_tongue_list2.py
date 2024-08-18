import os
import cv2
from alignment_V3 import align_segmented_tongue

def process_images(input_folder, output_folder, output_with_points_folder, original_with_points_folder):
    for folder in [output_folder, output_with_points_folder, original_with_points_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    successful_count = 0
    total_count = 0

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                total_count += 1
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path, file)
                output_with_points_path = os.path.join(output_with_points_folder, relative_path, f"points_{file}")
                original_with_points_path = os.path.join(original_with_points_folder, relative_path, f"original_points_{file}")
                
                for path in [output_path, output_with_points_path, original_with_points_path]:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                
                try:
                    color_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
                    if color_image is None:
                        print(f"Failed to load image: {input_path}")
                        continue

                    aligned_image, original_tip, original_top, aligned_tip, aligned_top, _, _ = align_segmented_tongue(color_image)
                    
                    # 创建带有关键点的对齐图像
                    aligned_with_points = aligned_image.copy()
                    cv2.circle(aligned_with_points, aligned_tip, 5, (0, 0, 255), -1)  # 舌尖用红色标记
                    cv2.circle(aligned_with_points, aligned_top, 5, (255, 255, 0), -1)  # 舌顶用青色标记
                    cv2.line(aligned_with_points, aligned_tip, aligned_top, (255, 0, 255), 2)  # 连线用粉色标记

                    # 创建带有关键点的原始图像
                    original_with_points = color_image.copy()
                    cv2.circle(original_with_points, original_tip, 5, (0, 0, 255), -1)  # 舌尖用红色标记
                    cv2.circle(original_with_points, original_top, 5, (255, 255, 0), -1)  # 舌顶用青色标记
                    cv2.line(original_with_points, original_tip, original_top, (255, 0, 255), 2)  # 连线用粉色标记

                    # 保存图像
                    cv2.imwrite(output_path, aligned_image)
                    cv2.imwrite(output_with_points_path, aligned_with_points)
                    cv2.imwrite(original_with_points_path, original_with_points)
                    
                    successful_count += 1
                    if successful_count % 100 == 0:
                        print(f"Processed {successful_count} images successfully out of {total_count} total images.")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

    print(f"Processing completed. Total images: {total_count}, Successfully processed: {successful_count}")

if __name__ == "__main__":
    input_folder = "seg_cropped_images"  # 替换为您的输入文件夹路径
    output_folder = "seg_cropped_images_align"  # 替换为您想要保存处理后图像的文件夹路径
    output_with_points_folder = "seg_cropped_images_align_with_points"  # 替换为您想要保存带关键点图像的文件夹路径
    original_with_points_folder = "seg_cropped_images_original_with_points"  # 替换为您想要保存带关键点原始图像的文件夹路径
    
    process_images(input_folder, output_folder, output_with_points_folder, original_with_points_folder)