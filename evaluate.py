import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# Cấu hình đường dẫn
# ==========================================
OUTPUT_PATH = "output"
X_PATH = os.path.join(OUTPUT_PATH, "X_train.npy")
Y_PATH = os.path.join(OUTPUT_PATH, "y_train.npy")
MODEL_PATH = os.path.join(OUTPUT_PATH, "cnn_model.h5")
CLASSES = ["negative", "positive"]

def evaluate():
    # 1. Kiểm tra file mô hình
    if not os.path.exists(MODEL_PATH):
        print("LỖI: Chưa tìm thấy file cnn_model.h5! Hãy chạy train.py trước.")
        return

    # 2. Tải dữ liệu và mô hình
    print("--- 1. Đang tải mô hình và dữ liệu ---")
    model = tf.keras.models.load_model(MODEL_PATH)
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # 3. Dự đoán kết quả
    print("--- 2. Đang tiến hành đánh giá ---")
    y_pred_probs = model.predict(X)
    
    # Chuyển xác suất (0.0 -> 1.0) thành nhãn (0 hoặc 1)
    y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

    # 4. In Báo cáo Đánh giá (Classification Report)
    print("\n" + "="*45)
    print("          BÁO CÁO ĐÁNH GIÁ (REPORT)")
    print("="*45)
    print(classification_report(y, y_pred, target_names=CLASSES))

    # 5. In Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y, y_pred)
    print("="*45)
    print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print("="*45)
    print(f"| Thực tế \\ Dự đoán | Negative (0) | Positive (1) |")
    print(f"| Negative (0)     | {cm[0][0]:<12} | {cm[0][1]:<12} |")
    if len(cm) > 1:
        print(f"| Positive (1)     | {cm[1][0]:<12} | {cm[1][1]:<12} |")
    print("="*45)

if __name__ == "__main__":
    evaluate()