import os
import glob
import numpy as np
import librosa

# ==========================================
# Cấu hình đường dẫn & Tham số (Hyperparameters)
# ==========================================
DATASET_PATH = "dataset"
OUTPUT_PATH = "output"
CLASSES = ["negative", "positive"] # 0: negative, 1: positive

# Các tham số trích xuất âm thanh
SAMPLE_RATE = 16000    # Tần số lấy mẫu 16kHz
DURATION = 1           # Độ dài file âm thanh (1 giây)
N_MFCC = 13            # Số lượng hệ số MFCC
N_FFT = 2048           # Độ dài cửa sổ FFT
HOP_LENGTH = 512       # Bước nhảy cửa sổ

# Số lượng mẫu chuẩn cho 1 file âm thanh 1s
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_mfcc(file_path):
    """
    Hàm đọc file WAV và trích xuất đặc trưng MFCC
    """
    try:
        # 1. Đọc file WAV
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # 2. Xử lý độ dài: Padding nếu thiếu, cắt bỏ nếu thừa
        if len(signal) < SAMPLES_PER_TRACK:
            signal = librosa.util.fix_length(signal, size=SAMPLES_PER_TRACK)
        else:
            signal = signal[:SAMPLES_PER_TRACK]
            
        # 3. Trích xuất MFCC
        mfcc = librosa.feature.mfcc(
            y=signal, 
            sr=sr, 
            n_mfcc=N_MFCC, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
        
        # Transpose thành (time_steps, N_MFCC) -> Dạng matrix theo thời gian
        return mfcc.T

    except Exception as e:
        print(f"Lỗi file {file_path}: {e}")
        return None

def process_dataset():
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    X = []
    y = []

    print("--- BẮT ĐẦU TRÍCH XUẤT MFCC ---")
    
    for label_idx, label_name in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_PATH, label_name)
        # Quét tất cả file .wav trong dataset/negative hoặc dataset/positive
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        print(f"-> Đang xử lý '{label_name}' (Nhãn: {label_idx}) | Tìm thấy {len(wav_files)} file .wav")
        
        for file_path in wav_files:
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                X.append(mfcc)
                y.append(label_idx)

    # Chuyển đổi list thành NumPy Array
    X = np.array(X)
    y = np.array(y)

    # Thêm 1 chiều channel ở cuối (dạng ảnh 1 channel cho CNN 2D): (samples, steps, mfcc, 1)
    X = X[..., np.newaxis]

    print("\n--- HOÀN THÀNH ---")
    print(f"Shape của X: {X.shape}") # Ví dụ: (Số_file, 32, 13, 1)
    print(f"Shape của y: {y.shape}")

    # Đường dẫn lưu file npy vào thư mục output
    x_save_path = os.path.join(OUTPUT_PATH, "X_train.npy")
    y_save_path = os.path.join(OUTPUT_PATH, "y_train.npy")

    # Lưu dữ liệu
    np.save(x_save_path, X)
    np.save(y_save_path, y)
    print(f"Đã xuất dữ liệu thành công vào thư mục '{OUTPUT_PATH}/'!")

if __name__ == "__main__":
    process_dataset()