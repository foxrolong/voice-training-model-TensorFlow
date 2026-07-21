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

BATCH_SIZE = 16
EPOCHS = 35

def build_model(input_shape):
    """
    Mô hình CNN được tối ưu lại cho ma trận MFCC nhỏ (32, 13, 1)
    """
    model = models.Sequential([
        # Khai báo Input rõ ràng cho Keras 3
        layers.Input(shape=input_shape),

        # Khối Conv 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Khối Conv 2 (Giảm MaxPool xuống 2x1 để không bóp nghẹt chiều MFCC)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Khối Conv 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Chuyển đổi thành vector 1D
        layers.Flatten(),
        
        # Dense Layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        
        # Output: Sigmoid phân loại nhị phân
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("--- 1. Đang tải dữ liệu từ thư mục 'output/' ---")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("LỖI: Không tìm thấy file X_train.npy hoặc y_train.npy! Hãy chạy preprocess.py trước.")
        return

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Đã tải thành công! Shape X: {X.shape}, Shape y: {y.shape}")

    # Chia tập Train / Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Khởi tạo mô hình
    input_shape = X_train.shape[1:] # (32, 13, 1)
    model = build_model(input_shape)
    model.summary()

    # Huấn luyện mô hình
    print("\n--- 2. Bắt đầu huấn luyện mô hình CNN ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Lưu mô hình
    model.save(MODEL_SAVE_PATH)
    print(f"\n--- HOÀN THÀNH ---")
    print(f"Mô hình đã được lưu tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()