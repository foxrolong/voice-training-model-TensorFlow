import os
import shutil

# Thư mục chứa file wav sau khi convert
SOURCE_DIR = "output"

# Thư mục dataset
DATASET_DIR = "dataset"

# Số file positive đầu tiên
POSITIVE_COUNT = 40


positive_dir = os.path.join(DATASET_DIR, "positive")
negative_dir = os.path.join(DATASET_DIR, "negative")

os.makedirs(positive_dir, exist_ok=True)
os.makedirs(negative_dir, exist_ok=True)


# Lấy danh sách file wav
files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(".wav")
]

# Sắp xếp theo tên
files.sort()


for index, file in enumerate(files):

    src = os.path.join(SOURCE_DIR, file)

    if index < POSITIVE_COUNT:
        dst = os.path.join(positive_dir, file)
        label = "positive"
    else:
        dst = os.path.join(negative_dir, file)
        label = "negative"


    shutil.copy(src, dst)

    print(
        f"{file} -> {label}"
    )


print("\nĐã sắp xếp xong!")