import cv2
import os
import numpy as np
from natsort import natsorted

def create_video_from_images(image_folder, output_video_path, fps=30):
 
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsorted(images)  

    if not images:
        print("No PNG images found in the specified folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

    print(f"Video created successfully: {output_video_path}")

image_folder = "final_combine"  
output_video = "output_video_2fps.mp4"  
fps = 2  

create_video_from_images(image_folder, output_video, fps)