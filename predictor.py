import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

    
def pre_process_image2(img):
    """
    Tiền xử lý ảnh để chuẩn bị cho việc nhận diện chữ số/chữ cái
    Tối ưu cho ảnh scan/photocopy có nền trắng và chữ đen
    """
    # Tạo bản copy để không thay đổi ảnh gốc
    img_copy = img.copy()
    
    # Kiểm tra nếu ảnh là màu thì chuyển về grayscale
    if len(img_copy.shape) == 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng Gaussian blur nhẹ để giảm noise
    img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
    
    # Sử dụng adaptive threshold thay vì OTSU cho ảnh scan
    # THRESH_BINARY (không phải INV) vì chữ đen trên nền trắng
    img_binary = cv2.adaptiveThreshold(
        img_copy, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,  # INV để chữ trở thành trắng trên nền đen
        11, 
        2
    )
    
    # Áp dụng morphological operations để làm sạch
    kernel = np.ones((2,2), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    
    # Tìm contour để crop chính xác vùng chứa chữ
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Lọc các contour có diện tích quá nhỏ
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        
        if valid_contours:
            # Tìm bounding box chung của tất cả contours hợp lệ
            all_points = np.vstack([cnt.reshape(-1, 2) for cnt in valid_contours])
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Crop vùng chứa chữ với một chút margin
            margin = 2
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_binary.shape[1], x + w + margin)
            y2 = min(img_binary.shape[0], y + h + margin)
            
            digit_img = img_binary[y1:y2, x1:x2]
        else:
            digit_img = img_binary
    else:
        digit_img = img_binary
    
    # Kiểm tra nếu ảnh quá nhỏ hoặc trống
    if digit_img.shape[0] < 3 or digit_img.shape[1] < 3:
        # Tạo ảnh trống 28x28
        digit_img = np.zeros((28, 28), dtype=np.float32)
        return digit_img
    
    # Thêm padding để tạo khoảng trắng xung quanh chữ
    # Tính toán padding để tạo ảnh vuông
    h, w = digit_img.shape
    if h > w:
        pad_w = (h - w) // 2
        pad_h = 4
        digit_img = cv2.copyMakeBorder(digit_img, pad_h, pad_h, pad_w, pad_w + (h - w) % 2, 
                                       cv2.BORDER_CONSTANT, value=0)
    else:
        pad_h = (w - h) // 2
        pad_w = 4
        digit_img = cv2.copyMakeBorder(digit_img, pad_h, pad_h + (w - h) % 2, pad_w, pad_w, 
                                       cv2.BORDER_CONSTANT, value=0)
    
    # Resize về 28x28 với interpolation phù hợp
    digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values về [0, 1]
    digit_img = digit_img.astype(np.float32) / 255.0
    
    return digit_img


def predict2(image_array, model, show_plot=False):
    """
    Dự đoán ký tự A, B, C, D từ ảnh đã được tiền xử lý
    
    Args:
        image_array: Ảnh đã được tiền xử lý (28x28) hoặc (1, 28, 28) hoặc (28, 28, 1)
        model: Model đã được train
        show_plot: Có hiển thị biểu đồ không (mặc định False cho web app)
    
    Returns:
        predicted_char: Ký tự dự đoán ('A', 'B', 'C', 'D' hoặc 'EMPTY')
    """
    
    # Chuẩn hóa shape của ảnh đầu vào
    if len(image_array.shape) == 2:  # (28, 28)
        image_input = image_array.reshape(1, 28, 28, 1)
    elif len(image_array.shape) == 3:  # (28, 28, 1) hoặc (1, 28, 28)
        if image_array.shape[0] == 1:  # (1, 28, 28)
            image_input = image_array.reshape(1, 28, 28, 1)
        else:  # (28, 28, 1)
            image_input = image_array.reshape(1, 28, 28, 1)
    elif len(image_array.shape) == 4:  # (1, 28, 28, 1)
        image_input = image_array
    else:
        raise ValueError(f"Unsupported image shape: {image_array.shape}")
    
    # Kiểm tra nếu ảnh quá trống (ít pixel khác 0)
    non_zero_pixels = np.count_nonzero(image_array)
    total_pixels = image_array.size
    
    if non_zero_pixels < total_pixels * 0.02:  # Nếu ít hơn 2% pixel có giá trị
        return 'EMPTY'
    
    try:
        # Dự đoán từ model
        predictions = model.predict(image_input, verbose=0)
        
        # Tùy thuộc vào loại model, có thể cần điều chỉnh mapping
        # Option 1: Nếu model của bạn được train với labels 0,1,2,3 cho A,B,C,D
        if predictions.shape[1] == 4:  # Model chỉ có 4 classes (A,B,C,D)
            class_labels = ['A', 'B', 'C', 'D']
            predicted_index = np.argmax(predictions[0])
            predicted_char = class_labels[predicted_index]
            confidence = predictions[0][predicted_index]
            
        # Option 2: Nếu model là EMNIST với 26 classes (A-Z)
        elif predictions.shape[1] >= 26:
            # EMNIST mapping: A=0, B=1, C=2, D=3, ...
            limit_letters = [0, 1, 2, 3]  # Indices cho A, B, C, D
            limited_probs = predictions[0][limit_letters]
            best_index_in_limited = np.argmax(limited_probs)
            predicted_index = limit_letters[best_index_in_limited]
            predicted_char = chr(predicted_index + ord('A'))
            confidence = limited_probs[best_index_in_limited]
            
        # Option 3: Model khác (tự điều chỉnh)
        else:
            # Giả sử model có output khác, cần mapping riêng
            predicted_index = np.argmax(predictions[0])
            # Thêm logic mapping tùy theo model của bạn
            if predicted_index < 4:
                predicted_char = chr(predicted_index + ord('A'))
                confidence = predictions[0][predicted_index]
            else:
                predicted_char = 'UNKNOWN'
                confidence = 0.0
        
        # Kiểm tra confidence threshold
        if confidence < 0.3:  # Nếu độ tin cậy quá thấp
            predicted_char = 'UNCERTAIN'
        
        # Hiển thị kết quả chỉ khi được yêu cầu (cho debugging)
        if show_plot:
            plt.figure(figsize=(6, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image_array.squeeze(), cmap='gray')
            plt.title(f"Dự đoán: {predicted_char}\nConfidence: {confidence:.3f}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            if predictions.shape[1] == 4:
                labels = ['A', 'B', 'C', 'D']
                plt.bar(labels, predictions[0])
            else:
                plt.bar(['A', 'B', 'C', 'D'], limited_probs)
            plt.title('Confidence Scores')
            plt.ylabel('Probability')
            plt.tight_layout()
            plt.show()
        
        return predicted_char
        
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return 'ERROR'


def detect_table_structure(img, thresh):
    """
    Phát hiện cấu trúc bảng bằng morphological operations
    """
    horizal = thresh.copy()
    vertical = thresh.copy()

    scale_height = 15
    scale_long = 15

    long = int(img.shape[1] / scale_long)
    height = int(img.shape[0] / scale_height)

    # Tạo structural elements cho đường ngang
    horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
    horizal = cv2.erode(horizal, horizalStructure)
    horizal = cv2.dilate(horizal, horizalStructure)

    # Tạo structural elements cho đường dọc
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # Tạo mask tổng hợp
    mask = cv2.bitwise_or(vertical, horizal)
    
    return mask


def find_largest_table(mask):
    """
    Tìm bảng có diện tích lớn nhất trong ảnh
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    x_max, y_max, w_max, h_max = 0, 0, 0, 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            x_max, y_max, w_max, h_max = x, y, w, h
            max_area = area
    
    return x_max, y_max, w_max, h_max


def extract_table_cells(img, thresh, x_max, y_max, w_max, h_max, num_rows=19, start_row=1):
    """
    Trích xuất các ô từ bảng đã được phát hiện
    """
    cropped_thresh_img = []
    cropped_origin_img = []
    contours_img = []
    
    # Trích xuất các ô từ cột đầu tiên
    for i in range(start_row, num_rows):
        row_start = y_max + round(i * h_max / num_rows)
        row_end = y_max + round((i + 1) * h_max / num_rows)
        col_start = x_max + round(w_max / 6)
        col_end = x_max + round(w_max / 2)
        
        thresh1 = thresh[row_start:row_end, col_start:col_end]
        contours_thresh1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        origin1 = img[row_start:row_end, col_start:col_end]

        cropped_thresh_img.append(thresh1)
        cropped_origin_img.append(origin1)
        contours_img.append(contours_thresh1)

    # Trích xuất các ô từ cột thứ hai
    for i in range(start_row, num_rows):
        row_start = y_max + round(i * h_max / num_rows)
        row_end = y_max + round((i + 1) * h_max / num_rows)
        col_start = x_max + round(2 * w_max / 3)
        col_end = x_max + round(w_max)
        
        thresh1 = thresh[row_start:row_end, col_start:col_end]
        contours_thresh1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        origin1 = img[row_start:row_end, col_start:col_end]

        cropped_thresh_img.append(thresh1)
        cropped_origin_img.append(origin1)
        contours_img.append(contours_thresh1)
    
    return cropped_thresh_img, cropped_origin_img, contours_img


def extract_character_from_cell(cell_img, contours):
    """
    Trích xuất ký tự từ một ô cụ thể
    """
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            
            img_height, img_width = cell_img.shape
            
            # Kiểm tra vị trí và kích thước hợp lý
            if (w > 3 and h > 3 and
                x > img_width * 0.02 and x < img_width * 0.98 and
                y > img_height * 0.02 and y < img_height * 0.98):
                
                # Trích xuất vùng chứa ký tự với margin
                margin = 2
                x1 = max(0, x - margin)
                y1 = max(0, y - margin) 
                x2 = min(img_width, x + w + margin)
                y2 = min(img_height, y + h + margin)
                
                character_img = cell_img[y1:y2, x1:x2]
                
                if character_img.shape[0] > 3 and character_img.shape[1] > 3:
                    return character_img
    
    return None


def process_all_cells(cropped_thresh_img, cropped_origin_img, contours_img, model, verbose=False):
    """
    Xử lý tất cả các ô và nhận diện ký tự
    
    Args:
        cropped_thresh_img: Danh sách ảnh threshold
        cropped_origin_img: Danh sách ảnh gốc
        contours_img: Danh sách contours
        model: Model AI
        verbose: Có in chi tiết không (mặc định False cho web app)
    
    Returns:
        results: Danh sách kết quả nhận diện
    """
    results = []
    
    for i, contour_img in enumerate(contours_img):
        if verbose:
            print(f"\n=== Xử lý Cell {i} ===")
        
        try:
            # Trích xuất ký tự từ ô
            character_img = extract_character_from_cell(cropped_origin_img[i], contour_img)
            
            # Nếu không tìm thấy ký tự rõ ràng, sử dụng toàn bộ ô
            if character_img is None:
                if verbose:
                    print(f"Không tìm thấy ký tự rõ ràng trong Cell {i}, xử lý toàn bộ cell")
                character_img = cropped_origin_img[i]
            
            # Tiền xử lý ảnh
            processed_img = pre_process_image2(character_img)
            
            # In thông tin chi tiết
            if verbose:
                print(f"Processed image shape: {processed_img.shape}")
                print(f"Pixel value range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
                print(f"Non-zero pixels: {np.count_nonzero(processed_img)}/{processed_img.size}")
            
            # Dự đoán với function đã fix
            if np.count_nonzero(processed_img) > 5:  # Giảm threshold từ 10 xuống 5
                char = predict2(processed_img, model, show_plot=False)
                results.append(char)
                if verbose:
                    print(f"--> Đã nhận diện: {char} - Cell {i}")
            else:
                results.append("EMPTY")
                if verbose:
                    print(f"--> Cell trống - Cell {i}")
            
        except Exception as e:
            if verbose:
                print(f"Lỗi xử lý Cell {i}: {e}")
                import traceback
                traceback.print_exc()
            results.append("ERROR")
        
        if verbose:
            print("-" * 60)
    
    return results


def predict_from_opencv_image(opencv_image, model):
    """
    Dự đoán từ opencv image (thay vì file path)
    Hàm chính để xử lý ảnh từ web app
    """
    try:
        # Chuyển sang grayscale
        if len(opencv_image.shape) == 3:
            img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        else:
            img = opencv_image
            
        # Áp dụng các bước xử lý tương tự như trong code gốc
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect table structure
        mask = detect_table_structure(img, thresh)
        x_max, y_max, w_max, h_max = find_largest_table(mask)
        
        # Extract cells
        cropped_thresh_img, cropped_origin_img, contours_img = extract_table_cells(
            img, thresh, x_max, y_max, w_max, h_max
        )
        
        # Process all cells (không verbose để tránh spam log)
        results = process_all_cells(cropped_thresh_img, cropped_origin_img, contours_img, model, verbose=False)
        
        return results
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        import traceback
        traceback.print_exc()
        return []


# Hàm để test độc lập (có thể comment khi không cần)
def main_test(image_path="tn.jpg", model_path="emnist_cnn_model.h5"):
    """
    Hàm test cho việc phát triển và debug
    """
    try:
        # Load image
        img = cv2.imread(image_path, 0)
        if img is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return []
            
        # Load model
        model = load_model(model_path)
        
        # Convert to opencv format
        opencv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Predict
        results = predict_from_opencv_image(opencv_image, model)
        
        print(f"Tổng số chữ số được trích xuất: {len(results)}")
        print(f"Kết quả nhận diện: {results}")
        
        return results
        
    except Exception as e:
        print(f"Lỗi trong main_test: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # Test function - chỉ chạy khi file được thực thi trực tiếp
    results = main_test()