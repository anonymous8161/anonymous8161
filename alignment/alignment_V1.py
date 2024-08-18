import cv2
import numpy as np
import math

def find_tongue_contours(gray_image):
    height, width = gray_image.shape[:2]
    _, binary = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    upper_contour = []
    lower_contour = []
    
    for x in range(width):
        col = binary[:, x]
        non_zero = np.nonzero(col)[0]
        if len(non_zero) > 0:
            upper_y = non_zero[0]
            lower_y = non_zero[-1]
            upper_contour.append((x, upper_y))
            lower_contour.append((x, lower_y))
    
    upper_contour = filter_steep_points(upper_contour)
    lower_contour = filter_steep_points(lower_contour)
    
    return np.array(upper_contour), np.array(lower_contour)

def filter_steep_points(contour):
    if len(contour) < 3:
        return contour
    filtered = [contour[0]] 
    for i in range(1, len(contour)):
        current_point = contour[i]
        prev_point = filtered[-1]
        
        dx = current_point[0] - prev_point[0]
        dy = current_point[1] - prev_point[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        
        if angle < 45:
            filtered.append(current_point)
    
    return filtered

def smooth_contour(contour, gray_image, smoothness=10):
    if len(contour) < 3:
        return contour  
    x = [p[0] for p in contour]
    y = [p[1] for p in contour]
    
    num_points = max(len(contour), 3)  
    x_smooth = np.linspace(min(x), max(x), num_points)
    y_smooth = np.interp(x_smooth, x, y)
    
    window = np.ones(min(smoothness, len(y_smooth))) / min(smoothness, len(y_smooth))
    y_smooth = np.convolve(y_smooth, window, mode='same')
    
    _, binary = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    height, width = binary.shape[:2]
    adjusted_contour = []
    for i in range(len(x_smooth)):
        x = int(x_smooth[i])
        y = int(y_smooth[i])
        if 0 <= x < width:
            col = binary[:, x]
            non_zero = np.nonzero(col)[0]
            if len(non_zero) > 0:
                if y < height // 2:  
                    y = non_zero[0]
                else: 
                    y = non_zero[-1]
        adjusted_contour.append((x, y))
    
    return np.array(adjusted_contour)

def find_tongue_tip_and_top(upper_smooth, lower_smooth):
    tongue_tip = tuple(lower_smooth[len(lower_smooth)//2])
    tongue_top = tuple(upper_smooth[len(upper_smooth)//2])
    return tongue_tip, tongue_top

def align_segmented_tongue(color_image, margin=5):
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    upper_contour, lower_contour = find_tongue_contours(gray_image)
    
    upper_smooth = smooth_contour(upper_contour, gray_image)
    lower_smooth = smooth_contour(lower_contour, gray_image)

    tongue_tip, tongue_top = find_tongue_tip_and_top(upper_smooth, lower_smooth)

    height, width = color_image.shape[:2]
    center_x, center_y = width // 2, height // 2

    dx = tongue_tip[0] - tongue_top[0]
    dy = tongue_tip[1] - tongue_top[1]
    angle = np.degrees(np.arctan2(dx, dy))

    diagonal = np.sqrt(width**2 + height**2)
    new_size = int(np.ceil(diagonal)) + 2 * margin  

    expanded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    expanded_image[y_offset:y_offset+height, x_offset:x_offset+width] = color_image

    new_center = (new_size // 2, new_size // 2)

    rotation_matrix = cv2.getRotationMatrix2D(new_center, -angle, 1.0)
    rotated = cv2.warpAffine(expanded_image, rotation_matrix, (new_size, new_size), flags=cv2.INTER_LINEAR)

    original_points = np.array([[tongue_tip[0] + x_offset, tongue_tip[1] + y_offset, 1],
                                [tongue_top[0] + x_offset, tongue_top[1] + y_offset, 1]])
    rotated_points = np.dot(rotation_matrix, original_points.T).T
    rotated_tip = tuple(map(int, rotated_points[0][:2]))
    rotated_top = tuple(map(int, rotated_points[1][:2]))

    mid_point = ((rotated_tip[0] + rotated_top[0]) / 2, (rotated_tip[1] + rotated_top[1]) / 2)
    dx = new_center[0] - mid_point[0]
    dy = 0 

    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(rotated, translation_matrix, (new_size, new_size))

    aligned_tip = (int(rotated_tip[0] + dx), int(rotated_tip[1] + dy))
    aligned_top = (int(rotated_top[0] + dx), int(rotated_top[1] + dy))

    crop_x = (new_size - width) // 2 - margin
    crop_y = (new_size - height) // 2 - margin
    aligned_cropped = aligned[crop_y:crop_y+height+2*margin, crop_x:crop_x+width+2*margin]

    aligned_tip = (aligned_tip[0] - crop_x, aligned_tip[1] - crop_y)
    aligned_top = (aligned_top[0] - crop_x, aligned_top[1] - crop_y)

    return aligned_cropped, tongue_tip, tongue_top, aligned_tip, aligned_top, upper_smooth, lower_smooth

def crop_aligned_image(aligned_image, margin=5):
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    non_zero = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(non_zero)
    
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(aligned_image.shape[1] - x, w + 2 * margin)
    h = min(aligned_image.shape[0] - y, h + 2 * margin)
    
    cropped = aligned_image[y:y+h, x:x+w]
    
    return cropped

if __name__ == "__main__":
    color_image = cv2.imread('36.jpg', cv2.IMREAD_COLOR)
    if color_image is None:
        raise ValueError("Failed to load the image. Please check the file path.")

    aligned_image, original_tip, original_top, aligned_tip, aligned_top, upper_smooth, lower_smooth = align_segmented_tongue(color_image)

    cropped_aligned_image = crop_aligned_image(aligned_image)

    original_with_points = color_image.copy()
    cv2.circle(original_with_points, original_tip, 5, (0, 0, 255), -1)  
    cv2.circle(original_with_points, original_top, 5, (255, 255, 0), -1) 
    cv2.line(original_with_points, original_tip, original_top, (255, 0, 255), 2)  

    for point in upper_smooth:
        cv2.circle(original_with_points, tuple(point), 1, (255, 0, 0), -1)  
    for point in lower_smooth:
        cv2.circle(original_with_points, tuple(point), 1, (0, 255, 255), -1)  

    aligned_with_points = aligned_image.copy()
    cv2.circle(aligned_with_points, aligned_tip, 5, (0, 0, 255), -1) 
    cv2.circle(aligned_with_points, aligned_top, 5, (255, 255, 0), -1)  
    cv2.line(aligned_with_points, aligned_tip, aligned_top, (255, 0, 255), 2)  

    cropped_with_points = cropped_aligned_image.copy()

    crop_offset_x = aligned_image.shape[1] // 2 - cropped_aligned_image.shape[1] // 2
    crop_offset_y = aligned_image.shape[0] // 2 - cropped_aligned_image.shape[0] // 2
    cropped_tip = (aligned_tip[0] - crop_offset_x, aligned_tip[1] - crop_offset_y)
    cropped_top = (aligned_top[0] - crop_offset_x, aligned_top[1] - crop_offset_y)
    cv2.circle(cropped_with_points, cropped_tip, 5, (0, 0, 255), -1)  
    cv2.circle(cropped_with_points, cropped_top, 5, (255, 255, 0), -1)  
    cv2.line(cropped_with_points, cropped_tip, cropped_top, (255, 0, 255), 2) 

    cv2.imshow('Original Tongue with Key Points', original_with_points)
    cv2.imshow('Aligned Tongue with Key Points', aligned_with_points)
    cv2.imshow('Cropped Aligned Tongue with Key Points', cropped_with_points)

    cv2.waitKey(0)
    cv2.destroyAllWindows()