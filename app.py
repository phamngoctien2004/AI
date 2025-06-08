import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
from io import BytesIO
import zipfile
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Import các hàm từ code gốc của bạn
# Giả sử bạn đã tạo file predictor.py chứa các hàm cần thiết

def load_student_codes(uploaded_file):
    """Đọc file mã sinh viên"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Chỉ hỗ trợ file CSV hoặc Excel")
            return None
        
        # Kiểm tra cột cần thiết
        if 'ma_sv' not in df.columns:
            st.error("File phải có cột 'ma_sv' chứa mã sinh viên")
            return None
            
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def process_uploaded_images(uploaded_files):
    """Xử lý các file ảnh được upload"""
    images_dict = {}
    for uploaded_file in uploaded_files:
        # Lấy tên file (không có extension)
        filename = os.path.splitext(uploaded_file.name)[0]
        
        # Chuyển đổi uploaded file thành opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        images_dict[filename] = opencv_image
    
    return images_dict

def predict_from_opencv_image(opencv_image, model):
    """Dự đoán từ opencv image (thay vì file path)"""
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
        
        # Detect table structure (sử dụng các hàm từ code gốc)
        mask = detect_table_structure(img, thresh)
        x_max, y_max, w_max, h_max = find_largest_table(mask)
        
        # Extract cells
        cropped_thresh_img, cropped_origin_img, contours_img = extract_table_cells(
            img, thresh, x_max, y_max, w_max, h_max
        )
        
        # Process all cells
        results = process_all_cells(cropped_thresh_img, cropped_origin_img, contours_img, model)
        
        return results
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return []

def calculate_score(predicted_answers, correct_answers, total_questions=36):
    """Tính điểm dựa trên đáp án đúng"""
    if len(predicted_answers) != len(correct_answers):
        st.warning(f"Số câu trả lời ({len(predicted_answers)}) không khớp với đáp án ({len(correct_answers)})")
    
    correct_count = 0
    for i in range(min(len(predicted_answers), len(correct_answers))):
        if predicted_answers[i] == correct_answers[i]:
            correct_count += 1
    
    score = (correct_count / total_questions) * 10
    return score, correct_count

def display_grading_result(student_id, predicted_answers, correct_answers, score, correct_count):
    """Hiển thị kết quả chấm bài chi tiết"""
    st.subheader(f"Kết quả chấm bài - Mã SV: {student_id}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Điểm số", f"{score:.1f}/10")
    with col2:
        st.metric("Số câu đúng", f"{correct_count}/{len(correct_answers)}")
    with col3:
        st.metric("Tỷ lệ đúng", f"{(correct_count/len(correct_answers)*100):.1f}%")
    
    # Hiển thị chi tiết từng câu
    with st.expander("Xem chi tiết từng câu"):
        df_detail = pd.DataFrame({
            'Câu': range(1, len(correct_answers) + 1),
            'Đáp án đúng': correct_answers,
            'Trả lời': predicted_answers[:len(correct_answers)] if len(predicted_answers) >= len(correct_answers) 
                      else predicted_answers + [''] * (len(correct_answers) - len(predicted_answers)),
            'Kết quả': ['✓' if i < len(predicted_answers) and predicted_answers[i] == correct_answers[i] 
                       else '✗' for i in range(len(correct_answers))]
        })
        st.dataframe(df_detail, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Hệ thống chấm bài tự động",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("🎯 Hệ thống chấm bài tự động")
    st.markdown("---")
    
    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Load model
        model_file = st.file_uploader("Tải model AI", type=['h5'], key="model")
        if model_file:
            try:
                # Lưu model tạm thời
                with open("temp_model.h5", "wb") as f:
                    f.write(model_file.getbuffer())
                model = load_model("temp_model.h5")
                st.success("✅ Model đã được tải thành công!")
            except Exception as e:
                st.error(f"❌ Lỗi tải model: {e}")
                model = None
        else:
            # Thử load model mặc định
            try:
                model = load_model("emnist_cnn_model.h5")
                st.info("📋 Sử dụng model mặc định")
            except:
                model = None
                st.warning("⚠️ Chưa có model AI")
        
        # Nhập đáp án đúng
        st.subheader("📋 Đáp án chuẩn")
        answer_input_method = st.radio(
            "Cách nhập đáp án:",
            ["Nhập thủ công", "Upload file"]
        )
        
        if answer_input_method == "Nhập thủ công":
            correct_answers_str = st.text_area(
                "Nhập đáp án (cách nhau bởi dấu phẩy)",
                placeholder="A,B,C,D,A,B,C,D..."
            )
            if correct_answers_str:
                correct_answers = [ans.strip().upper() for ans in correct_answers_str.split(',')]
                st.success(f"✅ Đã nhập {len(correct_answers)} đáp án")
            else:
                correct_answers = []
        else:
            answer_file = st.file_uploader("Upload file đáp án", type=['txt', 'csv'])
            if answer_file:
                content = answer_file.read().decode('utf-8')
                correct_answers = [ans.strip().upper() for ans in content.replace('\n', ',').split(',') if ans.strip()]
                st.success(f"✅ Đã tải {len(correct_answers)} đáp án")
            else:
                correct_answers = []
    
    # Main content
    if model is None:
        st.error("❌ Vui lòng tải model AI trước khi sử dụng")
        return
    
    if not correct_answers:
        st.warning("⚠️ Vui lòng nhập đáp án chuẩn")
        return
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📁 Tải dữ liệu", "🔍 Chấm bài", "📊 Kết quả"])
    
    with tab1:
        st.header("📁 Tải dữ liệu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Danh sách mã sinh viên")
            student_file = st.file_uploader(
                "Tải file danh sách sinh viên (CSV/Excel)",
                type=['csv', 'xlsx'],
                help="File phải có cột 'ma_sv' chứa mã sinh viên"
            )
            
            if student_file:
                df_students = load_student_codes(student_file)
                if df_students is not None:
                    st.success(f"✅ Đã tải {len(df_students)} sinh viên")
                    st.dataframe(df_students.head(), use_container_width=True)
                    
                    # Lưu vào session state
                    st.session_state['students'] = df_students
        
        with col2:
            st.subheader("2. Ảnh bài làm")
            uploaded_images = st.file_uploader(
                "Tải ảnh bài làm",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Tên file ảnh phải trùng với mã sinh viên"
            )
            
            if uploaded_images:
                images_dict = process_uploaded_images(uploaded_images)
                st.success(f"✅ Đã tải {len(images_dict)} ảnh")
                
                # Hiển thị preview
                with st.expander("Xem trước ảnh"):
                    for filename, img in list(images_dict.items())[:3]:  # Chỉ hiển thị 3 ảnh đầu
                        st.text(f"📷 {filename}")
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, width=300)
                
                # Lưu vào session state
                st.session_state['images'] = images_dict
    
    with tab2:
        st.header("🔍 Chấm bài tự động")
        
        if 'students' not in st.session_state or 'images' not in st.session_state:
            st.warning("⚠️ Vui lòng tải dữ liệu ở tab 'Tải dữ liệu' trước")
            return
        
        df_students = st.session_state['students']
        images_dict = st.session_state['images']
        
        st.info(f"📊 Có {len(df_students)} sinh viên và {len(images_dict)} ảnh")
        
        if st.button("🚀 Bắt đầu chấm bài", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, row in df_students.iterrows():
                student_id = str(row['ma_sv'])
                progress = (idx + 1) / len(df_students)
                progress_bar.progress(progress)
                status_text.text(f"Đang chấm bài sinh viên: {student_id} ({idx + 1}/{len(df_students)})")
                
                if student_id in images_dict:
                    # Predict answers
                    predicted_answers = predict_from_opencv_image(images_dict[student_id], model)
                    
                    # Calculate score
                    score, correct_count = calculate_score(predicted_answers, correct_answers)
                    
                    results.append({
                        'ma_sv': student_id,
                        'diem': score,
                        'so_cau_dung': correct_count,
                        'tong_cau': len(correct_answers),
                        'dap_an_du_doan': ','.join(predicted_answers),
                        'trang_thai': 'Đã chấm'
                    })
                else:
                    results.append({
                        'ma_sv': student_id,
                        'diem': 0,
                        'so_cau_dung': 0,
                        'tong_cau': len(correct_answers),
                        'dap_an_du_doan': '',
                        'trang_thai': 'Không có ảnh'
                    })
            
            progress_bar.progress(1.0)
            status_text.text("✅ Hoàn thành chấm bài!")
            
            # Lưu kết quả
            st.session_state['results'] = pd.DataFrame(results)
            st.success(f"🎉 Đã chấm xong {len(results)} bài!")
    
    with tab3:
        st.header("📊 Kết quả chấm bài")
        
        if 'results' not in st.session_state:
            st.info("🔄 Chưa có kết quả. Vui lòng chấm bài trước.")
            return
        
        df_results = st.session_state['results']
        
        # Thống kê tổng quan
        st.subheader("📈 Thống kê tổng quan")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số bài", len(df_results))
        with col2:
            avg_score = df_results[df_results['trang_thai'] == 'Đã chấm']['diem'].mean()
            st.metric("Điểm trung bình", f"{avg_score:.2f}")
        with col3:
            pass_count = len(df_results[df_results['diem'] >= 5])
            st.metric("Số bài đạt", pass_count)
        with col4:
            pass_rate = (pass_count / len(df_results)) * 100
            st.metric("Tỷ lệ đạt", f"{pass_rate:.1f}%")
        
        # Bảng kết quả chi tiết
        st.subheader("📋 Bảng điểm chi tiết")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_status = st.selectbox("Lọc theo trạng thái", ["Tất cả", "Đã chấm", "Không có ảnh"])
        with col2:
            search_student = st.text_input("Tìm mã sinh viên")
        
        # Apply filters
        filtered_df = df_results.copy()
        if filter_status != "Tất cả":
            filtered_df = filtered_df[filtered_df['trang_thai'] == filter_status]
        if search_student:
            filtered_df = filtered_df[filtered_df['ma_sv'].str.contains(search_student, case=False)]
        
        # Display results
        st.dataframe(
            filtered_df[['ma_sv', 'diem', 'so_cau_dung', 'tong_cau', 'trang_thai']].style.format({
                'diem': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # Chi tiết từng sinh viên
        st.subheader("🔍 Xem chi tiết từng sinh viên")
        selected_student = st.selectbox(
            "Chọn sinh viên:",
            options=df_results['ma_sv'].tolist()
        )
        
        if selected_student:
            student_data = df_results[df_results['ma_sv'] == selected_student].iloc[0]
            if student_data['trang_thai'] == 'Đã chấm':
                predicted_answers = student_data['dap_an_du_doan'].split(',')
                display_grading_result(
                    selected_student,
                    predicted_answers,
                    correct_answers,
                    student_data['diem'],
                    student_data['so_cau_dung']
                )
            else:
                st.warning(f"Sinh viên {selected_student} chưa được chấm bài")
        
        # Download results
        st.subheader("💾 Tải kết quả")
        csv = df_results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Tải file CSV",
            data=csv,
            file_name="ket_qua_cham_bai.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()