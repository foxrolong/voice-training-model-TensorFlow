<h1 align="center">Quy trình huấn luyện mô hình nhận giọng bằng TensorFlow cho ESP32-S3</h1>

## Mục tiêu

Xây dựng một mô hình AI có khả năng nhận đúng câu lệnh tích hợp vào ESP32-S3 "đánh thức".

## Bước 1: Thu thập dữ liệu

Ghi âm các câu lệnh bằng chính giọng của bạn/người dùng.

* **Positive**: Chỉ chứa câu lệnh cần nhận, ví dụ:

  * "Mở đèn"
  * Thu khoảng 500–1000-10000 mẫu.

* **Negative**: Các câu nói khác như:

  * "Tắt đèn"
  * "Xin chào"
  * "Một hai ba"
  * "Hello"
  * Thu khoảng 500–1000-10000 mẫu.

* **Noise**: Âm thanh môi trường:

  * Tiếng quạt
  * Tiếng xe
  * Tiếng TV
  * Tiếng gió
  * Thu khoảng 500–1000-10000 mẫu.

Tất cả file âm thanh nên có định dạng WAV, mono, 16 kHz, 16-bit PCM. Nếu bạn đã ghi bằng mp4 thì sử dụng [tool python](https://github.com/foxrolong/voice-training-model-TensorFlow/blob/main/convert-format.py) để chuyển định dạng về WAV, mono, 16 kHz, 16-bit.

> 💡 **LƯU Ý:**hãy thu thập dữ liệu thật nhiều để giọng nói được nhận diện một cách chính xác nhất.

## Bước 2: Tiền xử lý dữ liệu

Sử dụng Python để:

* Đọc file WAV.
* Chuẩn hóa âm lượng.
* Cắt hoặc đệm để tất cả có cùng thời lượng.
* Trích xuất đặc trưng MFCC (Mel Frequency Cepstral Coefficients).
* Chuyển dữ liệu thành tensor để đưa vào TensorFlow.

---

## Bước 3: Huấn luyện mô hình

Thiết kế một mô hình CNN nhỏ gồm:

* Input: Đặc trưng MFCC.
* Các lớp Convolution + ReLU.
* MaxPooling.
* Fully Connected.
* Softmax.

Mô hình sẽ học phân biệt:

* Đúng câu lệnh "Mở đèn" do người dùng nói.
* Các câu khác hoặc tiếng ồn.

---

## Bước 4: Đánh giá mô hình

Kiểm tra mô hình trên tập dữ liệu chưa từng huấn luyện.

Các chỉ số cần theo dõi:

* Accuracy
* Precision
* Recall
* False Positive Rate
* False Negative Rate

Mục tiêu là đạt độ chính xác trên 95% trong môi trường sử dụng thực tế.

---

## Bước 5: Chuyển đổi mô hình

Sau khi huấn luyện xong:

TensorFlow Model

↓

TensorFlow Lite (.tflite)

↓

Quantization (INT8)

↓

TensorFlow Lite Micro

Việc lượng tử hóa giúp giảm kích thước mô hình và tăng tốc độ suy luận trên ESP32-S3.

---

## Bước 6: Chạy trên ESP32-S3

ESP32-S3 sẽ thực hiện:

1. Đọc dữ liệu từ microphone I2S.
2. Chia tín hiệu thành các khung (frame).
3. Tính toán MFCC giống như khi huấn luyện.
4. Đưa dữ liệu vào mô hình TensorFlow Lite Micro.
5. Nhận kết quả xác suất.

Ví dụ:

* Xác suất > 0.95 → Nhận diện thành công.
* Xác suất ≤ 0.95 → Bỏ qua.

---

## Bước 7: Điều khiển thiết bị

Khi mô hình nhận đúng câu lệnh:

"Mở đèn"

↓

ESP32-S3 xuất mức HIGH lên GPIO.

↓

LED sáng.

Sau khi kiểm tra hoạt động ổn định, thay phần điều khiển LED bằng lệnh đánh thức Xiaozhi AI hoặc điều khiển SmartKey.

---

## Cấu trúc dự án

```text
coice-training-model-TensorFlow/
│
├── dataset/
│   ├── positive/
│   ├── negative/
│   └── noise/
│
├── train/
│   ├── record.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_tflite.py
│
└── model/
    ├── model.tflite
    └── model_data.cc

```

## Mục tiêu nhỏ

Hoàn thành dự án nhỏ có khả năng:

* Nhận đúng câu lệnh "Mở đèn" bằng chính giọng của người dùng.
* Chạy hoàn toàn offline trên ESP32-S3.
* Điều khiển LED theo thời gian thực.
