import cv2
import numpy as np
import math

TOP_LIMIT_RATIO = 5  
BOTTOM_LIMIT_RATIO = 20 

def find_adaptive_limits(gray_image):
    height, width = gray_image.shape[:2]
    _, binary = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    
    row_sums = np.sum(binary, axis=1)
    
    max_row = np.argmax(row_sums)
    
    top_limit = max_row
    bottom_limit = max_row
    threshold = row_sums[max_row] * 0.5  
    
    while top_limit > 0 and row_sums[top_limit] > threshold:
        top_limit -= 1
    
    while bottom_limit < height - 1 and row_sums[bottom_limit] > threshold:
        bottom_limit += 1
    
    top_ratio = height / (top_limit + 1)
    bottom_ratio = height / (height - bottom_limit + 1)
    
    return top_ratio, bottom_ratio

def find_tongue_contours(gray_image):
    global TOP_LIMIT_RATIO, BOTTOM_LIMIT_RATIO
    height, width = gray_image.shape[:2]
    _, binary = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    upper_contour = []
    lower_contour = []
    
    top_limit = height // int(TOP_LIMIT_RATIO)  # Double insurance
    bottom_limit = height - height // int(BOTTOM_LIMIT_RATIO)

    for x in range(width):
        col = binary[:, x]
        non_zero = np.nonzero(col)[0]
        if len(non_zero) > 0:
            upper_y = next((y for y in non_zero if y <= top_limit), None)
            lower_y = next((y for y in reversed(non_zero) if y >= bottom_limit), None)
            
            if upper_y is not None:
                upper_contour.append((x, upper_y))
            if lower_y is not None:
                lower_contour.append((x, lower_y))
    
    upper_contour = filter_steep_points(upper_contour, height)
    lower_contour = filter_steep_points(lower_contour, height)
    
    return np.array(upper_contour), np.array(lower_contour)

def filter_steep_points(contour, height):
    if len(contour) < 3:  
        return contour


    top_limit = height // int(TOP_LIMIT_RATIO)
    bottom_limit = height - height // int(BOTTOM_LIMIT_RATIO)


    sorted_contour = sorted(contour, key=lambda point: point[0])


    lowest_point = max((p for p in sorted_contour if p[1] >= bottom_limit), key=lambda p: p[1], default=None)

    filtered = [sorted_contour[0]]
    for i in range(1, len(sorted_contour)):
        current_point = sorted_contour[i]
        
        if current_point[1] < top_limit: 
            prev_point = filtered[-1]
        elif current_point[1] >= bottom_limit: 
            prev_point = lowest_point if lowest_point else filtered[-1]
        else:  
            prev_point = filtered[-1]
        
        dx = current_point[0] - prev_point[0]
        dy = current_point[1] - prev_point[1]
        
        if dx == 0:
            angle = 90
        else:
            angle = abs(math.degrees(math.atan2(dy, dx)))
        
        if angle <= 60:  # 45-70 is also acceptable
            filtered.append(current_point)
    
    return filtered

def smooth_contour(contour, gray_image, smoothness=10, is_upper=True):
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
    top_limit = height // TOP_LIMIT_RATIO
    bottom_limit = height - height // BOTTOM_LIMIT_RATIO
    adjusted_contour = []
    for i in range(len(x_smooth)):
        x = int(x_smooth[i])
        y = int(y_smooth[i])
        if 0 <= x < width:
            col = binary[:, x]
            non_zero = np.nonzero(col)[0]
            if len(non_zero) > 0:
                if is_upper:  
                    y = non_zero[0]
                    if y <= top_limit:
                        adjusted_contour.append((x, y))
                else:  
                    y = non_zero[-1]
                    if y >= bottom_limit:
                        adjusted_contour.append((x, y))
    
    return np.array(adjusted_contour)

def find_tongue_tip_and_top(upper_smooth, lower_smooth):
  
    upper_smooth = sorted(upper_smooth, key=lambda p: p[0])
    lower_smooth = sorted(lower_smooth, key=lambda p: p[0])

    tongue_tip = lower_smooth[len(lower_smooth) // 2]
    
    tongue_top = upper_smooth[len(upper_smooth) // 2]
    
    return tuple(tongue_tip), tuple(tongue_top)

def align_segmented_tongue(color_image, margin=5, extra_margin=5):

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    upper_contour, lower_contour = find_tongue_contours(gray_image)
    
    upper_smooth = smooth_contour(upper_contour, gray_image, is_upper=True)
    lower_smooth = smooth_contour(lower_contour, gray_image, is_upper=False)

    tongue_tip, tongue_top = find_tongue_tip_and_top(upper_smooth, lower_smooth)

    height, width = color_image.shape[:2]
    center_x, center_y = width // 2, height // 2

    dx = tongue_tip[0] - tongue_top[0]
    dy = tongue_tip[1] - tongue_top[1]
    angle = np.degrees(np.arctan2(dx, dy))

    diagonal = np.sqrt(width**2 + height**2)
    expansion_factor = 1.5 
    new_size = int(np.ceil(diagonal * expansion_factor)) + 2 * margin

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

    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_aligned, 1, 255, cv2.THRESH_BINARY)
    non_zero = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(non_zero)

    x = max(0, x - margin - extra_margin)
    y = max(0, y - margin - extra_margin)
    w = min(new_size - x, w + 2 * (margin + extra_margin))
    h = min(new_size - y, h + 2 * (margin + extra_margin))

    aligned_cropped = aligned[y:y+h, x:x+w]

    aligned_tip = (aligned_tip[0] - x, aligned_tip[1] - y)
    aligned_top = (aligned_top[0] - x, aligned_top[1] - y)

    return aligned_cropped, tongue_tip, tongue_top, aligned_tip, aligned_top, upper_smooth, lower_smooth

def crop_aligned_image(aligned_image, margin=10):
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    non_zero = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(non_zero)
    
    height, width = aligned_image.shape[:2]
    
    left_space = x
    right_space = width - (x + w)
    top_space = y
    bottom_space = height - (y + h)
    
    crop_left, crop_right, crop_top, crop_bottom = 0, 0, 0, 0
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    
    if left_space < margin:
        pad_left = margin - left_space
    elif left_space > margin:
        crop_left = left_space - margin
    
    if right_space < margin:
        pad_right = margin - right_space
    elif right_space > margin:
        crop_right = right_space - margin
    
    if top_space < margin:
        pad_top = margin - top_space
    elif top_space > margin:
        crop_top = top_space - margin
    
    if bottom_space < margin:
        pad_bottom = margin - bottom_space
    elif bottom_space > margin:
        crop_bottom = bottom_space - margin
    
    cropped = aligned_image[y - crop_top : y + h + crop_bottom,
                            x - crop_left : x + w + crop_right]
    
    final_image = cv2.copyMakeBorder(cropped,
                                     pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return final_image


if __name__ == "__main__":
    img_name = 'cropped_frame_0034.png'

    color_image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if color_image is None:
        raise ValueError("Failed to load the image. Please check the file path.")

    aligned_image, original_tip, original_top, aligned_tip, aligned_top, upper_smooth, lower_smooth = align_segmented_tongue(color_image)

    cropped_aligned_image = aligned_image

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

    cv2.imshow('Original Tongue with Key Points', original_with_points)
    cv2.imshow('Aligned Tongue with Key Points', aligned_with_points)
    cv2.imwrite("./output.jpg", cropped_aligned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()