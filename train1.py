import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==========================================
# Cấu hình đường dẫn & Tham số
# ==========================================
OUTPUT_PATH = "output"
X_PATH = os.path.join(OUTPUT_PATH, "X_train.npy")
Y_PATH = os.path.join(OUTPUT_PATH, "y_train.npy")
MODEL_SAVE_PATH = os.path.join(OUTPUT_PATH, "cnn_model.h5")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30

def build_model(input_shape):
    """
    Xây dựng kiến trúc mô hình CNN 2D cho bài toán phân loại âm thanh
    """
    model = models.Sequential([
        # Khối Conv 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Khối Conv 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Khối Conv 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Chuyển đổi ma trận thành vector 1D
        layers.Flatten(),
        
        # Dense Layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Đầu ra: Sigmoid cho phân loại Nhị phân (0: negative, 1: positive)
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 1. Đọc dữ liệu từ file .npy
    print("--- 1. Đang tải dữ liệu từ thư mục 'output/' ---")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("LỖI: Không tìm thấy file X_train.npy hoặc y_train.npy! Hãy chạy preprocess.py trước.")
        return

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Đã tải thành công! Shape X: {X.shape}, Shape y: {y.shape}")

    # 2. Chia tập Train / Validation / Test (80% train, 20% test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Khởi tạo mô hình
    input_shape = X_train.shape[1:] # (time_steps, mfcc_features, 1)
    model = build_model(input_shape)
    model.summary()

    # 4. Huấn luyện mô hình
    print("\n--- 2. Bắt đầu huấn luyện mô hình CNN ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 5. Lưu mô hình đã huấn luyện
    model.save(MODEL_SAVE_PATH)
    print(f"\n--- HOÀN THÀNH ---")
    print(f"Mô hình đã được lưu tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()