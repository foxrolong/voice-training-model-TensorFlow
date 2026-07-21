import os
import numpy as np
import tensorflow as tf

# ==========================================
# Đường dẫn
# ==========================================
OUTPUT_PATH = "output"
H5_MODEL_PATH = os.path.join(OUTPUT_PATH, "cnn_model.h5")
X_DATA_PATH = os.path.join(OUTPUT_PATH, "X_train.npy")
TFLITE_MODEL_PATH = os.path.join(OUTPUT_PATH, "model_int8.tflite")
C_HEADER_PATH = os.path.join(OUTPUT_PATH, "model_data.h")

def representative_dataset_gen():
    """Hàm cung cấp dữ liệu mẫu để TensorFlow căn chỉnh dải giá trị INT8"""
    X = np.load(X_DATA_PATH)
    # Lấy ngẫu nhiên khoảng 100 mẫu
    for i in range(min(100, len(X))):
        # Đảm bảo kiểu dữ liệu float32 và đúng shape (1, 32, 13, 1)
        sample = np.expand_dims(X[i], axis=0).astype(np.float32)
        yield [sample]

def convert_to_int8():
    print("--- Bắt đầu Quantization INT8 cho ESP32-S3 ---")
    
    if not os.path.exists(H5_MODEL_PATH):
        print("LỖI: Không tìm thấy file cnn_model.h5!")
        return

    # 1. Load mô hình Keras
    model = tf.keras.models.load_model(H5_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 2. Cấu hình Quantization INT8 hoàn toàn (Full Integer Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ép toàn bộ Input/Output và Ops về INT8 chuẩn TFLite Micro
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 3. Chuyển đổi
    tflite_quant_model = converter.convert()

    # 4. Lưu file .tflite
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_quant_model)
    print(f"-> Đã xuất file TFLite INT8: {TFLITE_MODEL_PATH} ({len(tflite_quant_model) / 1024:.2f} KB)")

    # 5. Xuất file C Header (model_data.h) cho ESP32-S3
    hex_array = [f"0x{b:02x}" for b in tflite_quant_model]
    
    c_header_content = f"""// File tự động tạo bởi export_tflite.py
// Mô hình đã Quantized INT8 cho ESP32-S3
// Kích thước: {len(tflite_quant_model)} bytes

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

// Căn lề 16-byte chuẩn TFLite Micro
alignas(16) const unsigned char g_model[] = {{
    {', '.join(hex_array)}
}};

const unsigned int g_model_len = {len(tflite_quant_model)};

#endif // MODEL_DATA_H
"""

    with open(C_HEADER_PATH, "w", encoding="utf-8") as f:
        f.write(c_header_content)

    print(f"-> HOÀN THÀNH! File 'model_data.h' đã sẵn sàng tại: {C_HEADER_PATH}")

if __name__ == "__main__":
    convert_to_int8()