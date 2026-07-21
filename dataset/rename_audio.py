import os

# Thư mục cần đổi tên
FOLDER = "negative"

# Tiền tố tên file
PREFIX = "negative-long"

files = [
    f for f in os.listdir(FOLDER)
    if f.endswith(".wav")
]

files.sort()

for i, file in enumerate(files, start=1):

    old_path = os.path.join(FOLDER, file)

    new_name = f"{PREFIX}_{i:03d}.wav"

    new_path = os.path.join(FOLDER, new_name)

    os.rename(old_path, new_path)

    print(f"{file} -> {new_name}")

print("Đổi tên xong!")