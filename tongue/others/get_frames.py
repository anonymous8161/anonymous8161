import cv2
import os

def extract_frames(video_path, output_folder, sample_rate=5):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error opening video file")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            
            cv2.imwrite(output_path, frame)
            
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames")
        
        frame_count += 1
    
    video.release()
    
    print(f"Extraction complete. Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_count}")


video_path = "VID_20240817_021359.mp4"  
output_folder = "images"

extract_frames(video_path, output_folder, sample_rate=3)